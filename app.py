# app.py - Enhanced with Portfolio Comparison
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from math import comb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📊", layout="wide")

# ============================================================================
# HELPER FUNCTIONS (EXISTING)
# ============================================================================

@st.cache_data(ttl=36000)  # Cache for 10 hours
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
    insufficient_data_tickers = [ticker for ticker in tickers if data[ticker].count() < 52]
    if insufficient_data_tickers:
        st.warning(f"⚠️ Insufficient data for: {', '.join(insufficient_data_tickers)}. They will be excluded.")
        data = data.drop(columns=insufficient_data_tickers)

    return data


def calculate_returns(data, frequency='weekly'):
    """Calculate returns based on frequency"""
    if frequency == 'weekly':
        data = data.resample('W-FRI').last()
    elif frequency == 'monthly':
        data = data.resample('M').last()
    
    returns = data.pct_change().dropna()
    return returns


@st.cache_data(ttl=3600)
def get_market_cap_weights(tickers):
    """Get market capitalization weights, fallback to equal weights if unavailable"""
    market_caps = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            market_cap = info.get('marketCap') or info.get('totalAssets')
            if market_cap:
                market_caps[ticker] = market_cap
        except:
            continue
    
    if not market_caps:
        return np.array([1 / len(tickers)] * len(tickers))
    
    total_cap = sum(market_caps.values())
    weights = np.array([market_caps.get(t, total_cap / len(tickers)) / total_cap for t in tickers])
    return weights / weights.sum()


def calculate_expected_returns(returns, Sigma_decimal, w_market, rf_rate, market_risk_premium, method='equilibrium'):
    """Calculate expected returns using equilibrium or historical method"""
    periods_per_year = 52 if st.session_state.frequency == 'weekly' else 12
    rf_rate_percent = rf_rate
    rf_decimal = rf_rate / 100
    
    if method == 'equilibrium':
        market_risk_premium_decimal = market_risk_premium / 100
        var_market_decimal = w_market @ Sigma_decimal @ w_market
        
        if var_market_decimal == 0:
            st.warning("Market portfolio has zero variance. Cannot calculate equilibrium returns.")
            return pd.Series([rf_rate_percent] * len(returns.columns), index=returns.columns)

        lambda_risk = market_risk_premium_decimal / var_market_decimal
        Pi_decimal = lambda_risk * (Sigma_decimal @ w_market)
        mu_decimal_total = Pi_decimal + rf_decimal
        return pd.Series(mu_decimal_total, index=returns.columns) * 100

    return returns.mean() * periods_per_year * 100


def black_litterman(Sigma_percent_sq, w_market, rf_rate, market_risk_premium, P, Q_percent, tau=0.025):
    """Apply Black-Litterman model"""
    Sigma_decimal = Sigma_percent_sq / 10000
    rf_decimal = rf_rate / 100
    market_risk_premium_decimal = market_risk_premium / 100
    var_market_decimal = w_market @ Sigma_decimal @ w_market
    
    if var_market_decimal == 0:
        st.warning("Cannot apply Black-Litterman: market portfolio has zero variance.")
        return None, None, None
    
    lambda_risk = market_risk_premium_decimal / var_market_decimal
    Pi_decimal = lambda_risk * (Sigma_decimal @ w_market)
    Q_decimal = Q_percent / 100
    
    tau_Sigma = tau * Sigma_decimal
    Omega = np.diag(np.diag(tau * P @ Sigma_decimal @ P.T))
    
    M_inv = np.linalg.inv(np.linalg.inv(tau_Sigma) + P.T @ np.linalg.inv(Omega) @ P)
    mu_bl_decimal = M_inv @ (np.linalg.inv(tau_Sigma) @ Pi_decimal + P.T @ np.linalg.inv(Omega) @ Q_decimal)
    mu_bl_percent = (mu_bl_decimal + rf_decimal) * 100
    
    Sigma_bl_decimal = Sigma_decimal + M_inv
    Sigma_bl_percent_sq = Sigma_bl_decimal * 10000
    
    pi_eq_percent = (Pi_decimal + rf_decimal) * 100
    
    return mu_bl_percent, Sigma_bl_percent_sq, pi_eq_percent


def optimize_portfolio(mu, Sigma, rf, allow_short=False):
    """Optimize portfolio for maximum Sharpe ratio"""
    n = len(mu)
    rf_decimal = rf / 100

    def neg_sharpe(w):
        w = w / np.sum(w)
        port_return_decimal = w @ mu / 100
        port_vol_decimal = np.sqrt(w @ Sigma @ w) / 100
        if port_vol_decimal == 0:
            return 1e9
        return -(port_return_decimal - rf_decimal) / port_vol_decimal
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n)) if not allow_short else None
    w0 = np.array([1/n] * n)
    
    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = result.x
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ Sigma @ weights)
        sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0
        return {'weights': weights, 'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}
    else:
        st.error(f"Optimization failed: {result.message}")
        return None


def allocate_risky_riskfree(optimal_portfolio, rf_percent, risk_aversion):
    """Determine allocation between optimal risky portfolio and risk-free asset"""
    mu_p_decimal = optimal_portfolio['return'] / 100
    sigma_p_decimal = optimal_portfolio['volatility'] / 100
    rf_decimal = rf_percent / 100
    sigma_p_sq_decimal = sigma_p_decimal ** 2
    
    if sigma_p_sq_decimal == 0 or risk_aversion == 0:
        alpha = 1.0 if mu_p_decimal > rf_decimal else 0.0
    else:
        excess_return_decimal = mu_p_decimal - rf_decimal
        alpha = excess_return_decimal / (risk_aversion * sigma_p_sq_decimal)
    
    alpha = np.clip(alpha, 0, 1)
    final_return_percent = (alpha * mu_p_decimal + (1 - alpha) * rf_decimal) * 100
    final_vol_percent = (alpha * sigma_p_decimal) * 100
    
    return {'alpha': alpha, 'final_return': final_return_percent, 'final_volatility': final_vol_percent}


def calculate_risk_metrics(returns, weights, confidence=0.95):
    """Calculate VaR, CVaR, and Max Drawdown"""
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, (1 - confidence) * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    return {'var_95': var * 100, 'cvar_95': cvar * 100, 'max_drawdown': max_drawdown * 100}


# ============================================================================
# NEW PORTFOLIO COMPARISON FUNCTIONS
# ============================================================================

def parse_portfolio_input(portfolio_text, all_available_tickers):
    """
    Parse user's actual portfolio input
    Format: TICKER WEIGHT (one per line)
    Example: 
    SPY 0.25
    AAPL 0.30
    BND 0.45
    """
    portfolio = {}
    lines = [line.strip() for line in portfolio_text.strip().split('\n') if line.strip()]
    
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            ticker = parts[0].upper()
            try:
                weight = float(parts[1])
                portfolio[ticker] = weight
            except ValueError:
                st.warning(f"Invalid weight for {ticker}: {parts[1]}")
                continue
    
    if not portfolio:
        return None
    
    # Normalize weights to sum to 1
    total = sum(portfolio.values())
    if total > 0:
        portfolio = {k: v/total for k, v in portfolio.items()}
    
    return portfolio


def expand_universe_and_data(optimal_tickers, actual_portfolio_tickers, start_date, end_date, frequency):
    """
    Download data for expanded universe (all unique assets)
    Returns: expanded_data, expanded_returns, all_tickers
    """
    all_tickers = list(set(optimal_tickers) | set(actual_portfolio_tickers))
    all_tickers.sort()
    
    # Download data for expanded universe
    expanded_data = download_data(all_tickers, start_date, end_date)
    if expanded_data is None:
        return None, None, None
    
    expanded_returns = calculate_returns(expanded_data, frequency)
    
    # Filter to only tickers that have data
    available_tickers = list(expanded_returns.columns)
    
    return expanded_data, expanded_returns, available_tickers


def align_portfolio_weights(portfolio_dict, all_tickers):
    """
    Convert portfolio dictionary to weight vector aligned with all_tickers
    Missing assets get 0 weight
    """
    weights = np.zeros(len(all_tickers))
    for i, ticker in enumerate(all_tickers):
        weights[i] = portfolio_dict.get(ticker, 0.0)
    return weights


def calculate_portfolio_characteristics(weights, mu, Sigma, rf_rate):
    """Calculate return, volatility, and Sharpe ratio for a portfolio"""
    port_return = weights @ mu
    port_vol = np.sqrt(weights @ Sigma @ weights)
    sharpe = (port_return - rf_rate) / port_vol if port_vol > 0 else 0
    
    return {
        'return': port_return,
        'volatility': port_vol,
        'sharpe': sharpe
    }


def calculate_deviation_metrics(w_optimal, w_actual, mu, Sigma, rf_rate):
    """
    Calculate comprehensive deviation metrics between optimal and actual portfolios
    """
    # Portfolio characteristics
    optimal_char = calculate_portfolio_characteristics(w_optimal, mu, Sigma, rf_rate)
    actual_char = calculate_portfolio_characteristics(w_actual, mu, Sigma, rf_rate)
    
    # Gaps
    return_gap = actual_char['return'] - optimal_char['return']
    vol_gap = actual_char['volatility'] - optimal_char['volatility']
    sharpe_gap = actual_char['sharpe'] - optimal_char['sharpe']
    
    # Tracking error (volatility of the difference)
    weight_diff = w_actual - w_optimal
    tracking_error = np.sqrt(weight_diff @ Sigma @ weight_diff)
    
    # Mahalanobis distance (correlation-adjusted distance)
    try:
        Sigma_inv = np.linalg.inv(Sigma)
        mahalanobis_dist = np.sqrt(weight_diff @ Sigma_inv @ weight_diff)
    except:
        mahalanobis_dist = None
    
    # Euclidean distance (simple L2 norm of weight differences)
    euclidean_dist = np.sqrt(np.sum(weight_diff ** 2))
    
    return {
        'return_gap': return_gap,
        'volatility_gap': vol_gap,
        'sharpe_gap': sharpe_gap,
        'tracking_error': tracking_error,
        'mahalanobis_distance': mahalanobis_dist,
        'euclidean_distance': euclidean_dist,
        'optimal': optimal_char,
        'actual': actual_char
    }


def analyze_weight_contributions(w_optimal, w_actual, asset_names, mu, Sigma):
    """
    Decompose weight differences and their impact on return and risk
    """
    weight_diff = w_actual - w_optimal
    
    # Return contribution of each weight difference
    return_contributions = weight_diff * mu
    
    # Risk contribution (marginal contribution to portfolio volatility)
    vol_actual = np.sqrt(w_actual @ Sigma @ w_actual)
    if vol_actual > 0:
        marginal_risk = (Sigma @ w_actual) / vol_actual
        vol_contributions = weight_diff * marginal_risk
    else:
        vol_contributions = np.zeros(len(asset_names))
    
    deviation_df = pd.DataFrame({
        'Asset': asset_names,
        'Optimal_Weight_%': w_optimal * 100,
        'Actual_Weight_%': w_actual * 100,
        'Difference_%': weight_diff * 100,
        'Return_Impact_%': return_contributions,
        'Risk_Impact_%': vol_contributions
    })
    
    # Sort by absolute difference
    deviation_df['Abs_Difference'] = np.abs(deviation_df['Difference_%'])
    deviation_df = deviation_df.sort_values('Abs_Difference', ascending=False)
    deviation_df = deviation_df.drop('Abs_Difference', axis=1)
    
    return deviation_df


def calculate_substitution_quality(actual_only_assets, optimal_assets, all_assets, Sigma_decimal, mu, w_optimal):
    """
    For assets in actual portfolio but not in optimal, measure quality as substitutes
    """
    if not actual_only_assets or not optimal_assets:
        return pd.DataFrame()
    
    # Convert to percentage units for correlation calculation
    Sigma = Sigma_decimal * 10000
    
    substitution_scores = {}
    
    for actual_asset in actual_only_assets:
        idx_actual = all_assets.index(actual_asset)
        
        # Get optimal portfolio assets indices
        optimal_indices = [all_assets.index(a) for a in optimal_assets if a in all_assets]
        
        if not optimal_indices:
            continue
        
        # Calculate weighted average correlation with optimal portfolio
        correlations = []
        for opt_idx in optimal_indices:
            # Correlation coefficient
            cov = Sigma[idx_actual, opt_idx]
            std_actual = np.sqrt(Sigma[idx_actual, idx_actual])
            std_opt = np.sqrt(Sigma[opt_idx, opt_idx])
            
            if std_actual > 0 and std_opt > 0:
                corr = cov / (std_actual * std_opt)
                weight_in_optimal = w_optimal[opt_idx]
                correlations.append(corr * weight_in_optimal)
        
        avg_correlation = sum(correlations) if correlations else 0
        
        # Return difference
        optimal_return = sum([w_optimal[all_assets.index(a)] * mu[all_assets.index(a)] 
                            for a in optimal_assets if a in all_assets])
        return_diff = mu[idx_actual] - optimal_return
        
        # Quality score: high correlation is good, similar return is good
        quality_score = avg_correlation - abs(return_diff) / 10
        
        substitution_scores[actual_asset] = {
            'Correlation_with_Optimal': avg_correlation,
            'Return_vs_Optimal_%': return_diff,
            'Quality_Score': quality_score
        }
    
    return pd.DataFrame(substitution_scores).T


def compute_overall_deviation_score(deviation_metrics):
    """
    Compute a single 0-100 deviation score with interpretation
    Lower is better (0 = perfect match)
    """
    # Component penalties
    sharpe_penalty = abs(deviation_metrics['sharpe_gap']) * 30
    tracking_penalty = deviation_metrics['tracking_error'] * 2
    return_penalty = abs(deviation_metrics['return_gap']) * 1.5
    vol_penalty = abs(deviation_metrics['volatility_gap']) * 1
    
    total_score = min(100, sharpe_penalty + tracking_penalty + return_penalty + vol_penalty)
    
    # Interpretation
    if total_score < 10:
        interpretation = "Excellent - Very close to optimal"
        color = "green"
    elif total_score < 25:
        interpretation = "Good - Minor deviations"
        color = "lightgreen"
    elif total_score < 50:
        interpretation = "Moderate - Noticeable deviations"
        color = "yellow"
    elif total_score < 75:
        interpretation = "Significant deviations from optimal"
        color = "orange"
    else:
        interpretation = "Poor - Major deviations from optimal"
        color = "red"
    
    return {
        'score': total_score,
        'interpretation': interpretation,
        'color': color,
        'breakdown': {
            'Sharpe_Penalty': sharpe_penalty,
            'Tracking_Penalty': tracking_penalty,
            'Return_Penalty': return_penalty,
            'Volatility_Penalty': vol_penalty
        }
    }


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("📊 Advanced Portfolio Optimizer")
st.markdown("*A scientifically-grounded tool using Modern Portfolio Theory & the Black-Litterman Model*")

# Session state initialization
if 'frequency' not in st.session_state:
    st.session_state.frequency = 'weekly'
if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False

with st.sidebar:
    st.header("⚙️ Configuration")
    tickers_input = st.text_area("Asset Tickers (one per line)", "SPY\nGLD\nCIBR\nMAGS", height=150)
    tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    years_back = col1.selectbox("Years of Data", [1, 2, 3, 5, 10], index=3)
    st.session_state.frequency = col2.selectbox("Frequency", ["weekly", "monthly"], index=0)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365.25)
    
    st.markdown("---")
    rf_rate = st.number_input("Risk-Free Rate (Annual %)", 0.0, 10.0, 3.0, 0.1)
    risk_aversion = st.slider("Risk Aversion (RA)", 1.0, 15.0, 5.0, 0.5, 
                             help="Higher RA means more conservative. 2-4: Aggressive, 4-6: Moderate, 6-10: Conservative, 10+: Very Conservative")
    
    st.markdown("---")
    returns_method = st.selectbox("Expected Returns Method", ["equilibrium", "historical"], index=0,
        help="Equilibrium: Market-implied returns. Historical: Simple average of past returns.")
    
    if returns_method == "equilibrium": 
        st.info("✅ **Using Equilibrium Returns**")
    
    use_custom_views = st.checkbox("Add Custom Black-Litterman Views", value=False)
    if use_custom_views:
        views_input = st.text_area("Views (format: TICKER1 TICKER2 VALUE)", 
                                   "SPY AGG 2.5", height=100,
                                   help="Each line: TICKER1 TICKER2 VALUE means TICKER1 outperforms TICKER2 by VALUE%")
    
    st.markdown("---")
    optimize_button = st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)

# Main content
if optimize_button:
    with st.spinner("📥 Downloading data and optimizing..."):
        try:
            data = download_data(tickers, start_date, end_date)
            if data is not None:
                returns = calculate_returns(data, st.session_state.frequency)
                tickers = list(returns.columns)
                
                periods_per_year = 52 if st.session_state.frequency == 'weekly' else 12
                Sigma_percent_sq = returns.cov() * periods_per_year * 10000
                Sigma_decimal = Sigma_percent_sq / 10000
                
                w_market = get_market_cap_weights(tickers)
                market_risk_premium = 6.0
                
                mu = calculate_expected_returns(returns, Sigma_decimal, w_market, rf_rate, market_risk_premium, method=returns_method)
                
                # Black-Litterman if requested
                if use_custom_views and views_input.strip():
                    P_list, Q_list = [], []
                    
                    def append_view(ticker_a, ticker_b, q_val):
                        p_row = np.zeros(len(tickers))
                        p_row[tickers.index(ticker_a)] = 1
                        p_row[tickers.index(ticker_b)] = -1
                        P_list.append(p_row)
                        Q_list.append(q_val)
                    
                    for view in views_input.strip().split('\n'):
                        parts = view.strip().split()
                        if len(parts) < 3:
                            continue
                        a, b, q_value = parts[0].upper(), parts[1].upper(), float(parts[2])
                        
                        if a in tickers and b in tickers:
                            append_view(a, b, q_value)
                    
                    if P_list:
                        P, Q = np.array(P_list), np.array(Q_list)
                        mu_bl, Sigma_bl, pi_eq = black_litterman(Sigma_percent_sq.values, w_market, rf_rate, market_risk_premium, P=P, Q_percent=Q)
                        mu = pd.Series(mu_bl, index=tickers)
                        Sigma_percent_sq = pd.DataFrame(Sigma_bl, index=tickers, columns=tickers)
                        st.success("✅ Black-Litterman model applied.")
                
                # Optimize
                opt_result = optimize_portfolio(mu.values, Sigma_percent_sq.values, rf_rate)
                
                if opt_result:
                    allocation = allocate_risky_riskfree(opt_result, rf_rate, risk_aversion)
                    risk_metrics = calculate_risk_metrics(returns, opt_result['weights'])
                    
                    # Store in session state for Portfolio Comparison tab
                    st.session_state.optimization_done = True
                    st.session_state.opt_result = opt_result
                    st.session_state.allocation = allocation
                    st.session_state.risk_metrics = risk_metrics
                    st.session_state.mu = mu
                    st.session_state.Sigma_percent_sq = Sigma_percent_sq
                    st.session_state.returns = returns
                    st.session_state.tickers = tickers
                    st.session_state.rf_rate = rf_rate
                    st.session_state.start_date = start_date
                    st.session_state.end_date = end_date
                    
                    st.success("✅ Optimization Complete!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Final Return", f"{allocation['final_return']:.2f}%")
                    col2.metric("Final Volatility", f"{allocation['final_volatility']:.2f}%")
                    sharpe = (allocation['final_return'] - rf_rate) / allocation['final_volatility'] if allocation['final_volatility'] > 0 else 0
                    col3.metric("Sharpe Ratio", f"{sharpe:.3f}")
                    col4.metric("Risky Allocation", f"{allocation['alpha']*100:.1f}%")
                    st.markdown("---")
                    
                    # Create tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Allocation", 
                        "📈 Efficient Frontier", 
                        "⚠️ Risk Analysis", 
                        "📋 Model Details",
                        "🔍 Portfolio Comparison"
                    ])
                    
                    with tab1:
                        st.subheader("Final Portfolio Allocation")
                        final_weights = {'Risk-Free': (1 - allocation['alpha'])}
                        risky_weights = allocation['alpha'] * opt_result['weights']
                        for i, ticker in enumerate(tickers):
                            if risky_weights[i] > 0.005:
                                final_weights[ticker] = risky_weights[i]
                        
                        final_df = pd.DataFrame(final_weights.items(), columns=['Asset', 'Weight']).sort_values('Weight', ascending=False)
                        final_df['Weight (%)'] = final_df['Weight'] * 100
                        
                        col1, col2 = st.columns([0.6, 0.4])
                        with col1:
                            fig = px.pie(final_df, values='Weight', names='Asset', title="Portfolio Composition", hole=0.3)
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            st.dataframe(final_df[['Asset', 'Weight (%)']].style.format({'Weight (%)': '{:.2f}%'}), 
                                       hide_index=True, use_container_width=True)
                    
                    with tab2:
                        st.subheader("Efficient Frontier & Capital Allocation Line")
                        m = 5
                        N = len(tickers)
                        n_portfolios = max(3000, comb(m + N - 1, N - 1))
                        rand_results = np.zeros((3, n_portfolios))
                        
                        for i in range(n_portfolios):
                            w = np.random.random(len(tickers))
                            w /= w.sum()
                            rand_results[0, i] = w @ mu.values
                            rand_results[1, i] = np.sqrt(w @ Sigma_percent_sq.values @ w)
                            rand_results[2, i] = (rand_results[0, i] - rf_rate) / rand_results[1, i]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=rand_results[1,:], y=rand_results[0,:], mode='markers',
                            name='Random Portfolios',
                            marker=dict(color=rand_results[2,:], showscale=True, 
                                      colorscale='viridis', size=5, 
                                      colorbar=dict(title="Sharpe Ratio"))
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[opt_result['volatility']], y=[opt_result['return']],
                            mode='markers', name='Optimal Risky Portfolio',
                            marker=dict(color='red', size=12, symbol='star')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[allocation['final_volatility']], y=[allocation['final_return']],
                            mode='markers', name='Your Portfolio',
                            marker=dict(color='green', size=12, symbol='diamond')
                        ))
                        
                        x_cal = np.linspace(0, opt_result['volatility'] * 1.5, 100)
                        y_cal = rf_rate + opt_result['sharpe'] * x_cal
                        fig.add_trace(go.Scatter(
                            x=x_cal, y=y_cal, mode='lines',
                            name='Capital Allocation Line',
                            line=dict(color='darkorange', width=2, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="Efficient Frontier",
                            xaxis_title="Volatility (Annual %)",
                            yaxis_title="Expected Return (Annual %)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        st.subheader("Downside Risk Analysis")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Value-at-Risk (95%)", f"{risk_metrics['var_95']:.2f}%")
                        col2.metric("Conditional VaR (95%)", f"{risk_metrics['cvar_95']:.2f}%")
                        col3.metric("Maximum Drawdown", f"{risk_metrics['max_drawdown']:.2f}%")
                    
                    with tab4:
                        st.subheader("Model Details")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Expected Returns (Annual %)**")
                            st.dataframe(mu.to_frame('Return').style.format('{:.2f}%'))
                        with col2:
                            st.markdown("**Volatility (Annual %)**")
                            vol_df = pd.DataFrame({
                                'Volatility': np.sqrt(np.diag(Sigma_percent_sq))
                            }, index=tickers)
                            st.dataframe(vol_df.style.format('{:.2f}%'))
                    
                    with tab5:
                        st.subheader("🔍 Portfolio Comparison")
                        st.markdown("""
                        Compare your actual portfolio with the optimal portfolio. Enter your current holdings below.
                        The system will evaluate deviations considering correlations, returns, and volatilities of all assets.
                        """)
                        
                        st.markdown("---")
                        
                        # Input for actual portfolio
                        st.markdown("### Enter Your Actual Portfolio")
                        actual_portfolio_input = st.text_area(
                            "Portfolio Holdings (format: TICKER WEIGHT, one per line)",
                            "SPY 0.25\nAAPL 0.05\nBND 0.70",
                            height=200,
                            help="Enter each asset on a new line with its weight (e.g., 'SPY 0.25' means 25% in SPY). Weights will be normalized to sum to 1."
                        )
                        
                        compare_button = st.button("📊 Compare Portfolios", type="primary")
                        
                        if compare_button and actual_portfolio_input.strip():
                            with st.spinner("Analyzing portfolio deviation..."):
                                # Parse actual portfolio
                                actual_portfolio = parse_portfolio_input(actual_portfolio_input, tickers)
                                
                                if actual_portfolio is None:
                                    st.error("Could not parse portfolio input. Please check the format.")
                                else:
                                    actual_tickers = list(actual_portfolio.keys())
                                    
                                    # Expand universe
                                    expanded_data, expanded_returns, all_tickers = expand_universe_and_data(
                                        tickers, actual_tickers, start_date, end_date, st.session_state.frequency
                                    )
                                    
                                    if expanded_returns is not None:
                                        # Recalculate covariance matrix for expanded universe
                                        periods_per_year = 52 if st.session_state.frequency == 'weekly' else 12
                                        expanded_Sigma_percent_sq = expanded_returns.cov() * periods_per_year * 10000
                                        expanded_Sigma_decimal = expanded_Sigma_percent_sq / 10000
                                        
                                        # Recalculate expected returns for expanded universe
                                        expanded_w_market = get_market_cap_weights(all_tickers)
                                        expanded_mu = calculate_expected_returns(
                                            expanded_returns, expanded_Sigma_decimal, 
                                            expanded_w_market, rf_rate, market_risk_premium, 
                                            method=returns_method
                                        )
                                        
                                        # Align weight vectors
                                        optimal_weights_dict = {tickers[i]: opt_result['weights'][i] * allocation['alpha'] 
                                                              for i in range(len(tickers))}
                                        optimal_weights_dict['Risk-Free'] = 1 - allocation['alpha']
                                        
                                        # Remove risk-free from comparison (focus on risky assets)
                                        w_optimal_aligned = align_portfolio_weights(optimal_weights_dict, all_tickers)
                                        w_actual_aligned = align_portfolio_weights(actual_portfolio, all_tickers)
                                        
                                        # Calculate deviation metrics
                                        deviation = calculate_deviation_metrics(
                                            w_optimal_aligned, w_actual_aligned,
                                            expanded_mu.values, expanded_Sigma_percent_sq.values, rf_rate
                                        )
                                        
                                        # Overall score
                                        overall_score = compute_overall_deviation_score(deviation)
                                        
                                        # Display results
                                        st.markdown("---")
                                        st.markdown("### 📈 Overall Deviation Score")
                                        
                                        col1, col2, col3 = st.columns([1, 2, 1])
                                        with col2:
                                            st.markdown(f"""
                                            <div style="text-align: center; padding: 20px; background-color: {overall_score['color']}; 
                                                        border-radius: 10px; margin: 10px 0;">
                                                <h1 style="margin: 0; font-size: 48px;">{overall_score['score']:.1f}</h1>
                                                <p style="margin: 5px 0; font-size: 18px;"><strong>{overall_score['interpretation']}</strong></p>
                                                <p style="margin: 0; font-size: 14px;">(0 = Perfect Match, 100 = Maximum Deviation)</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        st.markdown("---")
                                        st.markdown("### 📊 Performance Comparison")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric(
                                                "Expected Return Gap",
                                                f"{deviation['return_gap']:.2f}%",
                                                delta=f"{deviation['return_gap']:.2f}%",
                                                delta_color="normal"
                                            )
                                            st.caption(f"Optimal: {deviation['optimal']['return']:.2f}% | Actual: {deviation['actual']['return']:.2f}%")
                                        
                                        with col2:
                                            st.metric(
                                                "Volatility Gap",
                                                f"{deviation['volatility_gap']:.2f}%",
                                                delta=f"{deviation['volatility_gap']:.2f}%",
                                                delta_color="inverse"
                                            )
                                            st.caption(f"Optimal: {deviation['optimal']['volatility']:.2f}% | Actual: {deviation['actual']['volatility']:.2f}%")
                                        
                                        with col3:
                                            st.metric(
                                                "Sharpe Ratio Gap",
                                                f"{deviation['sharpe_gap']:.3f}",
                                                delta=f"{deviation['sharpe_gap']:.3f}",
                                                delta_color="normal"
                                            )
                                            st.caption(f"Optimal: {deviation['optimal']['sharpe']:.3f} | Actual: {deviation['actual']['sharpe']:.3f}")
                                        
                                        st.markdown("---")
                                        st.markdown("### 📏 Distance Metrics")
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.metric("Tracking Error", f"{deviation['tracking_error']:.2f}%",
                                                    help="Standard deviation of return differences between portfolios")
                                        
                                        with col2:
                                            if deviation['mahalanobis_distance'] is not None:
                                                st.metric("Mahalanobis Distance", f"{deviation['mahalanobis_distance']:.3f}",
                                                        help="Correlation-adjusted distance between portfolios")
                                        
                                        st.markdown("---")
                                        st.markdown("### 🔍 Weight Analysis")
                                        
                                        weight_analysis = analyze_weight_contributions(
                                            w_optimal_aligned, w_actual_aligned,
                                            all_tickers, expanded_mu.values,
                                            expanded_Sigma_percent_sq.values
                                        )
                                        
                                        # Show only assets with non-zero weights in either portfolio
                                        weight_analysis_filtered = weight_analysis[
                                            (weight_analysis['Optimal_Weight_%'] > 0.01) | 
                                            (weight_analysis['Actual_Weight_%'] > 0.01)
                                        ].copy()
                                        
                                        st.dataframe(
                                            weight_analysis_filtered.style.format({
                                                'Optimal_Weight_%': '{:.2f}%',
                                                'Actual_Weight_%': '{:.2f}%',
                                                'Difference_%': '{:.2f}%',
                                                'Return_Impact_%': '{:.2f}%',
                                                'Risk_Impact_%': '{:.3f}%'
                                            }).background_gradient(subset=['Difference_%'], cmap='RdYlGn_r'),
                                            use_container_width=True,
                                            height=400
                                        )
                                        
                                        st.markdown("---")
                                        st.markdown("### 💡 Substitution Analysis")
                                        st.caption("For assets in your portfolio that are NOT in the optimal portfolio")
                                        
                                        # Find assets only in actual portfolio
                                        optimal_assets_set = set([t for i, t in enumerate(all_tickers) 
                                                                if w_optimal_aligned[i] > 0.001])
                                        actual_only_assets = [t for i, t in enumerate(all_tickers) 
                                                            if w_actual_aligned[i] > 0.001 and t not in optimal_assets_set]
                                        
                                        if actual_only_assets:
                                            substitution_df = calculate_substitution_quality(
                                                actual_only_assets, list(optimal_assets_set),
                                                all_tickers, expanded_Sigma_decimal.values,
                                                expanded_mu.values, w_optimal_aligned
                                            )
                                            
                                            if not substitution_df.empty:
                                                st.dataframe(
                                                    substitution_df.style.format({
                                                        'Correlation_with_Optimal': '{:.3f}',
                                                        'Return_vs_Optimal_%': '{:.2f}%',
                                                        'Quality_Score': '{:.3f}'
                                                    }).background_gradient(subset=['Quality_Score'], cmap='RdYlGn'),
                                                    use_container_width=True
                                                )
                                                
                                                st.markdown("""
                                                **Interpretation:**
                                                - **Correlation**: Higher is better (asset moves similarly to optimal portfolio)
                                                - **Return vs Optimal**: Difference in expected returns
                                                - **Quality Score**: Combined metric (higher is better)
                                                """)
                                            else:
                                                st.info("No substitution analysis available.")
                                        else:
                                            st.info("All assets in your portfolio are also in the optimal portfolio.")
                                        
                                        st.markdown("---")
                                        st.markdown("### 🎯 Recommendations")
                                        
                                        # Generate recommendations based on analysis
                                        recommendations = []
                                        
                                        if abs(deviation['sharpe_gap']) > 0.5:
                                            if deviation['sharpe_gap'] < 0:
                                                recommendations.append("⚠️ Your portfolio has significantly lower risk-adjusted returns. Consider rebalancing towards the optimal allocation.")
                                            else:
                                                recommendations.append("✅ Your portfolio shows higher risk-adjusted returns, but verify this isn't due to increased risk.")
                                        
                                        if deviation['tracking_error'] > 5:
                                            recommendations.append("⚠️ High tracking error indicates substantial deviation from optimal. Review major weight differences.")
                                        
                                        if deviation['return_gap'] < -3:
                                            recommendations.append("⚠️ Your portfolio has notably lower expected returns. Consider increasing allocation to higher-return assets in the optimal portfolio.")
                                        
                                        if deviation['volatility_gap'] > 5:
                                            recommendations.append("⚠️ Your portfolio has significantly higher volatility. Consider diversifying or reducing exposure to volatile assets.")
                                        
                                        # Check for large overweights
                                        large_overweights = weight_analysis_filtered[weight_analysis_filtered['Difference_%'] > 10]
                                        if not large_overweights.empty:
                                            overweight_assets = ', '.join(large_overweights['Asset'].tolist()[:3])
                                            recommendations.append(f"📊 Large overweights detected in: {overweight_assets}. Consider rebalancing.")
                                        
                                        if not recommendations:
                                            recommendations.append("✅ Your portfolio is reasonably aligned with the optimal allocation.")
                                        
                                        for rec in recommendations:
                                            st.markdown(f"- {rec}")
                                        
                                    else:
                                        st.error("Could not download data for the expanded universe.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())

elif st.session_state.optimization_done:
    # Show saved results
    st.info("💡 Previous optimization results are displayed. Click 'Optimize Portfolio' to refresh.")
    
    allocation = st.session_state.allocation
    opt_result = st.session_state.opt_result
    risk_metrics = st.session_state.risk_metrics
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Return", f"{allocation['final_return']:.2f}%")
    col2.metric("Final Volatility", f"{allocation['final_volatility']:.2f}%")
    sharpe = (allocation['final_return'] - st.session_state.rf_rate) / allocation['final_volatility'] if allocation['final_volatility'] > 0 else 0
    col3.metric("Sharpe Ratio", f"{sharpe:.3f}")
    col4.metric("Risky Allocation", f"{allocation['alpha']*100:.1f}%")

else:
    st.info("👈 **Configure your portfolio in the sidebar and click 'Optimize Portfolio' to begin!**")
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Configure Assets")
        st.markdown("""
        - Enter stock/ETF tickers
        - Select data history
        - Choose frequency
        """)
    
    with col2:
        st.markdown("#### 2️⃣ Set Preferences")
        st.markdown("""
        - Risk-free rate
        - Risk aversion (RA)
        - Returns method
        """)
    
    with col3:
        st.markdown("#### 3️⃣ Optimize & Compare")
        st.markdown("""
        - Click Optimize
        - View results
        - Compare your portfolio
        """)

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only. Past performance is not indicative of future results.")
st.caption("Copyright (c) 2025, Omid Arhami. Licensed under the BSD 3-Clause License.")
