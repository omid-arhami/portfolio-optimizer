# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📊", layout="wide")

# ============================================================================
# HELPER FUNCTIONS (Corrected & Refined)
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_data(tickers, start_date, end_date):
    """Download historical data with robust error handling"""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    
    # Drop tickers with all NaN data (failed downloads)
    data.dropna(axis=1, how='all', inplace=True)
    
    # Forward-fill and back-fill to handle sporadic missing values
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    if data.empty:
        st.error("❌ No data could be downloaded for the selected tickers. Please check the symbols.")
        return None
        
    # Check for tickers with insufficient data points
    insufficient_data_tickers = [ticker for ticker in tickers if data[ticker].count() < 52] # Min 1 year of weekly data
    if insufficient_data_tickers:
        st.warning(f"⚠️ Insufficient data for: {', '.join(insufficient_data_tickers)}. They will be excluded.")
        data.drop(columns=insufficient_data_tickers, inplace=True)
        
    return data

@st.cache_data(ttl=3600)
def get_market_cap_weights(tickers):
    """Get market capitalization weights for tickers with error handling"""
    market_caps = {}
    valid_tickers = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            mc = info.get('marketCap')
            if mc:
                market_caps[ticker] = mc
                valid_tickers.append(ticker)
        except Exception:
            st.warning(f"Could not fetch market cap for {ticker}. It will be weighted equally.")
            # Assign an average market cap to avoid errors, or handle as you see fit
            market_caps[ticker] = np.mean(list(market_caps.values())) if market_caps else 1e10

    total_cap = sum(market_caps[t] for t in valid_tickers)
    weights = np.array([market_caps[t] / total_cap for t in valid_tickers])
    
    return weights, valid_tickers


def calculate_returns(prices, frequency='weekly'):
    """Calculate arithmetic returns at specified frequency"""
    if frequency == 'weekly':
        resample_rule = 'W-FRI' # Use Friday to ensure consistency
        prices_resampled = prices.resample(resample_rule).last()
    elif frequency == 'monthly':
        resample_rule = 'M'
        prices_resampled = prices.resample(resample_rule).last()
    else: # Daily
        prices_resampled = prices
    
    returns = prices_resampled.pct_change().dropna()
    return returns

# CRITICAL FIX: Corrected the calculation of equilibrium returns
def calculate_expected_returns(returns, Sigma_decimal, w_market, rf_rate_percent, method='equilibrium'):
    """
    Calculate expected returns. Equilibrium method is now theoretically sound.
    Returns values are in annual percentage.
    """
    periods_per_year = 52 if st.session_state.frequency == 'weekly' else 12
    
    if method == 'historical':
        # Historical average - simple but unstable
        mu = returns.mean() * periods_per_year * 100
        return mu
    
    elif method == 'equilibrium':
        # Market-implied equilibrium returns (from Black-Litterman)
        # All calculations must be in decimal form
        market_risk_premium_decimal = 0.06  # Assumed historical equity risk premium (6%)
        rf_decimal = rf_rate_percent / 100

        # Calculate market variance in decimal
        var_market_decimal = w_market @ Sigma_decimal @ w_market
        if var_market_decimal == 0:
            st.error("Market variance is zero. Cannot calculate equilibrium returns.")
            return pd.Series([rf_rate_percent] * len(returns.columns), index=returns.columns)

        # Implied risk aversion parameter (lambda or delta)
        lambda_risk = market_risk_premium_decimal / var_market_decimal
        
        # Implied equilibrium excess returns (Pi vector) in decimal
        Pi_decimal = lambda_risk * (Sigma_decimal @ w_market)
        
        # Add risk-free rate to get total expected returns
        mu_decimal_total = Pi_decimal + rf_decimal
        
        # Return as annualized percentage
        return pd.Series(mu_decimal_total, index=returns.columns) * 100
    
    # Fallback to historical if method is not recognized
    return returns.mean() * periods_per_year * 100

# CRITICAL FIX: Corrected the Black-Litterman model implementation
def black_litterman(Sigma_percent_sq, w_market, rf_percent, tau=0.025, P=None, Q_percent=None):
    """
    Black-Litterman model implementation.
    This function now correctly handles all unit conversions internally.
    """
    n = len(w_market)
    rf_decimal = rf_percent / 100
    Sigma_decimal = Sigma_percent_sq / 10000  # Convert from %-squared to decimal

    # 1. Reverse engineer equilibrium returns (in decimal)
    market_premium_decimal = 0.06
    var_market_decimal = w_market @ Sigma_decimal @ w_market
    lambda_risk = market_premium_decimal / var_market_decimal
    Pi = lambda_risk * (Sigma_decimal @ w_market)  # Decimal excess returns

    # Calculate Pi total returns for returning later
    Pi_total_percent = (Pi + rf_decimal) * 100
    
    if P is not None and Q_percent is not None:
        # 2. Incorporate views
        Q_decimal = Q_percent / 100  # Convert views from % to decimal
        
        # For model stability, Omega is assumed to be a diagonal matrix
        # representing the uncertainty of each view independently.
        omega_diag = np.diag(tau * P @ Sigma_decimal @ P.T)
        Omega = np.diag(omega_diag)
        
        Omega_inv = np.linalg.inv(Omega)
        tau_Sigma_inv = np.linalg.inv(tau * Sigma_decimal)
        
        # Posterior estimate of excess returns (in decimal)
        M_inv = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
        mu_BL_excess = M_inv @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q_decimal)
        
        # Add risk-free rate for total returns and convert to percentage
        mu_BL_total_percent = (mu_BL_excess + rf_decimal) * 100

        return mu_BL_total_percent, Sigma_percent_sq, Pi_total_percent

    # If no views, return the calculated equilibrium total returns in percentage
    return Pi_total_percent, Sigma_percent_sq, Pi_total_percent


def optimize_portfolio(mu, Sigma, rf, allow_short=False):
    """Optimize portfolio for maximum Sharpe ratio (Tangency Portfolio)"""
    n = len(mu)
    rf_decimal = rf / 100  # Ensure RF rate is in decimal for calculation

    # Objective function: minimize the negative Sharpe ratio
    def neg_sharpe(w):
        # Ensure weights sum to 1 to avoid division by zero issues
        w = w / np.sum(w)
        port_return_decimal = w @ mu / 100
        port_vol_decimal = np.sqrt(w @ Sigma @ w) / 100
        if port_vol_decimal == 0:
            return 1e9 # Return a large number if volatility is zero
        return -(port_return_decimal - rf_decimal) / port_vol_decimal
    
    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n)) if not allow_short else None
    
    w0 = np.array([1/n] * n) # Initial guess: equal weights
    
    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = result.x
        # Recalculate portfolio stats based on final weights in percentage terms
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ Sigma @ weights)
        sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0
        
        return {'weights': weights, 'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}
    else:
        st.error(f"Optimization failed: {result.message}")
        return None

def allocate_risky_riskfree(optimal_portfolio, rf_percent, risk_aversion):
    """Determine allocation between the optimal risky portfolio and the risk-free asset"""
    # Convert all inputs from percentage to decimal for calculation
    mu_p_decimal = optimal_portfolio['return'] / 100
    sigma_p_decimal = optimal_portfolio['volatility'] / 100
    rf_decimal = rf_percent / 100
    
    # Calculate variance in decimal
    sigma_p_sq_decimal = sigma_p_decimal ** 2
    
    # Avoid division by zero
    if sigma_p_sq_decimal == 0 or risk_aversion == 0:
        alpha = 1.0 if mu_p_decimal > rf_decimal else 0.0
    else:
        # Calculate optimal allocation to risky portfolio (alpha) using decimal values
        excess_return_decimal = mu_p_decimal - rf_decimal
        alpha = excess_return_decimal / (risk_aversion * sigma_p_sq_decimal)
    
    # We don't allow borrowing, so alpha is capped at 100%
    alpha = np.clip(alpha, 0, 1)
    
    # Calculate final portfolio stats and convert back to percentage for display
    final_return_percent = (alpha * mu_p_decimal + (1 - alpha) * rf_decimal) * 100
    final_vol_percent = (alpha * sigma_p_decimal) * 100
    
    return {'alpha': alpha, 'final_return': final_return_percent, 'final_volatility': final_vol_percent}


def calculate_risk_metrics(returns, weights, confidence=0.95):
    """Calculate VaR, CVaR, and Max Drawdown for the portfolio"""
    portfolio_returns = returns @ weights
    
    var = np.percentile(portfolio_returns, (1 - confidence) * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {'var_95': var * 100, 'cvar_95': cvar * 100, 'max_drawdown': max_drawdown * 100}

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("📊 Advanced Portfolio Optimizer")
st.markdown("*A scientifically-grounded tool using Modern Portfolio Theory & the Black-Litterman Model*")

# Store frequency in session state for access in functions
if 'frequency' not in st.session_state:
    st.session_state.frequency = 'weekly'

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers_input = st.text_area("Asset Tickers (one per line)", "SPY\nVTI\nGLD\nSFY\nCIBR\nMCHI\nNVDA\nAVGO\nAMZN\nGOOG", height=150)
    tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
    st.markdown("---")
    col1, col2 = st.columns(2)
    years_back = col1.selectbox("Years of Data", [1, 2, 3, 5, 10], index=3)
    st.session_state.frequency = col2.selectbox("Frequency", ["weekly", "monthly"], index=0)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365.25)
    st.markdown("---")
    rf_rate = st.number_input("Risk-Free Rate (Annual %)", 0.0, 10.0, 3.8, 0.1)
    risk_aversion = st.slider("Risk Aversion (λ)", 1.0, 15.0, 5.0, 0.5, help="Higher λ means more conservative. 2-4: Aggressive, 5-8: Moderate, 9+: Conservative")
    st.markdown("---")
    returns_method = st.selectbox("Expected Returns Method", ["equilibrium", "historical"], index=0, help="Equilibrium (recommended) is more stable and forward-looking than simple historical averages.")
    if returns_method == "equilibrium": st.info("✅ Using market-implied equilibrium returns.")
    else: st.warning("⚠️ Historical means are often unstable and poor predictors of future returns.")
    st.markdown("---")
    use_custom_views = st.checkbox("Add Custom Black-Litterman Views")
    if use_custom_views:
        st.markdown("**Your Views (e.g., 'SPY outperforms AGG by 2.5%')**")
        views_text = st.text_area("Views (one per line: TICKER1 TICKER2 VALUE)", "SPY AGG 2.5", height=100)
    st.markdown("---")
    optimize_button = st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)

if optimize_button:
    with st.spinner("Processing... This may take a moment."):
        try:
            prices = download_data(tickers, start_date, end_date)
            if prices is None or prices.empty: st.stop()
            
            tickers = list(prices.columns) # Update tickers to only valid ones
            returns = calculate_returns(prices, st.session_state.frequency)
            
            periods_per_year = 52 if st.session_state.frequency == 'weekly' else 12
            if len(returns) < periods_per_year:
                st.error(f"❌ Insufficient data. Need at least 1 year of {st.session_state.frequency} data.")
                st.stop()
            
            # Covariance matrix: in %-squared for interpretation, but converted to decimal for calculations
            Sigma_percent_sq = returns.cov() * periods_per_year * 10000
            Sigma_decimal = Sigma_percent_sq.values / 10000

            with st.spinner("Fetching market caps..."):
                w_market, valid_tickers_mc = get_market_cap_weights(tickers)
            if len(w_market) != len(tickers):
                st.warning("Could not fetch market cap for all assets. Using equal weights as fallback.")
                w_market = np.array([1/len(tickers)] * len(tickers))

            mu = calculate_expected_returns(returns, Sigma_decimal, w_market, rf_rate, returns_method)
            
            if use_custom_views and views_text.strip():
                views_lines = [line.strip() for line in views_text.split('\n') if line.strip()]
                P_list, Q_list = [], []
                for view in views_lines:
                    parts = view.split()
                    if len(parts) == 3 and parts[0] in tickers and parts[1] in tickers:
                        p_row = np.zeros(len(tickers))
                        p_row[tickers.index(parts[0])] = 1
                        p_row[tickers.index(parts[1])] = -1
                        P_list.append(p_row)
                        Q_list.append(float(parts[2]))
                if P_list:
                    P, Q = np.array(P_list), np.array(Q_list)
                    mu_bl, Sigma_bl, pi_eq = black_litterman(Sigma_percent_sq.values, w_market, rf_rate, P=P, Q_percent=Q)
                    mu = pd.Series(mu_bl, index=tickers)
                    Sigma_percent_sq = pd.DataFrame(Sigma_bl, index=tickers, columns=tickers)
                    st.success("✅ Black-Litterman model applied with your views.")

            opt_result = optimize_portfolio(mu.values, Sigma_percent_sq.values, rf_rate)
            
            if opt_result:
                allocation = allocate_risky_riskfree(opt_result, rf_rate, risk_aversion)
                risk_metrics = calculate_risk_metrics(returns, opt_result['weights'])
                st.success("✅ Optimization Complete!")
                
                # --- RESULTS ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Final Return", f"{allocation['final_return']:.2f}%")
                col2.metric("Final Volatility", f"{allocation['final_volatility']:.2f}%")
                sharpe = (allocation['final_return'] - rf_rate) / allocation['final_volatility'] if allocation['final_volatility'] > 0 else 0
                col3.metric("Sharpe Ratio", f"{sharpe:.3f}")
                col4.metric("Risky Allocation", f"{allocation['alpha']*100:.1f}%")
                st.markdown("---")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Allocation", "📈 Efficient Frontier", "⚠️ Risk Analysis", "📋 Model Details"])
                
                with tab1:
                    st.subheader("Final Portfolio Allocation")
                    final_weights = {'Risk-Free': (1 - allocation['alpha'])}
                    risky_weights = allocation['alpha'] * opt_result['weights']
                    for i, ticker in enumerate(tickers):
                        if risky_weights[i] > 0.005: final_weights[ticker] = risky_weights[i]
                    
                    final_df = pd.DataFrame(final_weights.items(), columns=['Asset', 'Weight']).sort_values('Weight', ascending=False)
                    final_df['Weight (%)'] = final_df['Weight'] * 100
                    
                    col1, col2 = st.columns([0.6, 0.4])
                    with col1:
                        fig = px.pie(final_df, values='Weight', names='Asset', title="Portfolio Composition", hole=0.3)
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.dataframe(final_df[['Asset', 'Weight (%)']].style.format({'Weight (%)': '{:.2f}%'}), hide_index=True, use_container_width=True)

                with tab2:
                    st.subheader("Efficient Frontier & Capital Allocation Line")
                    # Generate random portfolios for plotting
                    n_portfolios = 3000
                    rand_results = np.zeros((3, n_portfolios))
                    for i in range(n_portfolios):
                        w = np.random.random(len(tickers)); w /= w.sum()
                        rand_results[0, i] = w @ mu.values
                        rand_results[1, i] = np.sqrt(w @ Sigma_percent_sq.values @ w)
                        rand_results[2, i] = (rand_results[0, i] - rf_rate) / rand_results[1, i]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=rand_results[1,:], y=rand_results[0,:], mode='markers', name='Random Portfolios',
                        marker=dict(color=rand_results[2,:], showscale=True, colorscale='viridis', size=5, colorbar=dict(title="Sharpe Ratio"))))
                    
                    # Add Optimal Risky Portfolio (Tangency)
                    fig.add_trace(go.Scatter(x=[opt_result['volatility']], y=[opt_result['return']], mode='markers', name='Optimal Risky Portfolio',
                        marker=dict(color='red', size=12, symbol='star')))
                    
                    # Add Final User Portfolio
                    fig.add_trace(go.Scatter(x=[allocation['final_volatility']], y=[allocation['final_return']], mode='markers', name='Your Portfolio',
                        marker=dict(color='green', size=12, symbol='diamond')))

                    # Add Capital Allocation Line
                    x_cal = np.linspace(0, opt_result['volatility'] * 1.5, 100)
                    y_cal = rf_rate + opt_result['sharpe'] * x_cal
                    fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Capital Allocation Line', line=dict(color='darkorange', width=2, dash='dash')))

                    fig.update_layout(title="Efficient Frontier", xaxis_title="Volatility (Annual %)", yaxis_title="Expected Return (Annual %)", height=500, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.subheader("Downside Risk Analysis")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Value-at-Risk (95%)", f"{risk_metrics['var_95']:.2f}%", help=f"Your portfolio is not expected to lose more than {abs(risk_metrics['var_95']):.2f}% in a single {st.session_state.frequency} period, with 95% confidence.")
                    col2.metric("Conditional VaR (95%)", f"{risk_metrics['cvar_95']:.2f}%", help=f"In the worst 5% of scenarios, the average loss is expected to be {abs(risk_metrics['cvar_95']):.2f}%.")
                    col3.metric("Max Historical Drawdown", f"{risk_metrics['max_drawdown']:.2f}%", help="The largest peak-to-trough decline observed in the historical data.")
                
                with tab4:
                    st.subheader("Model Inputs & Outputs")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Expected Returns (Annual %)**")
                        st.dataframe(mu.to_frame('Return').style.format('{:.2f}%'), use_container_width=True)
                    with col2:
                        st.markdown("**Volatility (Annual %)**")
                        vol_df = pd.DataFrame({'Volatility': np.sqrt(np.diag(Sigma_percent_sq))}, index=tickers)
                        st.dataframe(vol_df.style.format('{:.2f}%'), use_container_width=True)
                    
                    st.markdown("**Correlation Matrix**")
                    fig = px.imshow(returns.corr(), text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r', title="Asset Correlation", zmin=-1, zmax=1)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("👈 **Configure your portfolio in the sidebar and click 'Optimize Portfolio' to begin!**")
    st.markdown("---")
    st.markdown("### How this Optimizer Works")
    st.markdown("""
    This tool finds the mathematically optimal way to allocate your wealth between a portfolio of risky assets (like stocks and ETFs) and a risk-free asset (like a savings account). Here’s the process:

    1.  **Data Collection**: It downloads historical price data for your chosen tickers from Yahoo Finance.
    2.  **Risk & Correlation**: It calculates the volatility (risk) of each asset and how they move in relation to one another (correlation).
    3.  **Expected Returns**: Instead of relying on noisy historical averages, it uses the **Black-Litterman model's** concept of *equilibrium returns*. This estimates the returns that the global market is implicitly expecting, creating a much more stable and logical starting point.
    4.  **Finding the Best Risky Portfolio**: Using Modern Portfolio Theory, it identifies the single "tangency portfolio" – the combination of your risky assets with the highest possible reward for the amount of risk taken (i.e., the highest Sharpe Ratio).
    5.  **Personalized Allocation**: Finally, it determines the ideal split between that best risky portfolio and the risk-free asset based on your personal **risk aversion**. A more conservative investor will hold more in the risk-free asset, while an aggressive one will hold more in the risky portfolio.
    """)

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only and does not constitute financial advice. Data is sourced from Yahoo Finance. Past performance is not indicative of future results.")

