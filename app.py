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
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_data(tickers, start_date, end_date):
    """Download historical data with robust error handling"""
    valid_tickers = []
    failed_tickers = []
    
    for ticker in tickers:
        try:
            test_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not test_data.empty and len(test_data) > 50:  # Require minimum data
                valid_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            failed_tickers.append(ticker)
    
    if failed_tickers:
        st.warning(f"⚠️ Failed to download or insufficient data: {', '.join(failed_tickers)}")
    
    if not valid_tickers:
        st.error("❌ No valid tickers found!")
        return None
    
    # Download all valid tickers
    data = yf.download(valid_tickers, start=start_date, end=end_date, progress=False)
    
    if len(valid_tickers) == 1:
        prices = data['Adj Close'].to_frame()
        prices.columns = valid_tickers
    else:
        prices = data['Adj Close']
    
    return prices

@st.cache_data(ttl=3600)
def get_market_cap_weights(tickers):
    """Get market capitalization weights for tickers"""
    market_caps = {}
    
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            market_caps[ticker] = info.get('marketCap', 1e9)  # Default if unavailable
        except:
            market_caps[ticker] = 1e9  # Default value
    
    total_cap = sum(market_caps.values())
    weights = np.array([market_caps[t] / total_cap for t in tickers])
    
    return weights

def calculate_returns(prices, frequency='weekly'):
    """Calculate arithmetic returns at specified frequency"""
    if frequency == 'weekly':
        prices_resampled = prices.resample('W').last()
    elif frequency == 'monthly':
        prices_resampled = prices.resample('M').last()
    else:
        prices_resampled = prices
    
    # Use arithmetic returns (not log returns) for consistency
    returns = prices_resampled.pct_change().dropna()
    return returns

def calculate_expected_returns(returns, Sigma, w_market, rf_rate, method='equilibrium'):
    """
    Calculate expected returns using different methods
    
    Methods:
    - 'historical': Sample mean (simple but noisy)
    - 'equilibrium': Market-implied returns (Black-Litterman without views) - DEFAULT
    """
    periods_per_year = 52  # Assuming weekly data
    
    if method == 'historical':
        # Historical average - simple but unstable
        mu = returns.mean() * periods_per_year * 100
        return mu
    
    elif method == 'equilibrium':
        # Equilibrium returns (what B-L uses) - much more stable
        market_risk_premium = 6.0  # Historical equity risk premium (%)
        var_market = w_market @ Sigma @ w_market
        lambda_risk = market_risk_premium / var_market
        Pi = lambda_risk * Sigma @ w_market
        return pd.Series(Pi, index=returns.columns)
    
    return returns.mean() * periods_per_year * 100

def black_litterman(Sigma, w_market, rf, tau=0.025, P=None, Q=None):
    """Black-Litterman model implementation"""
    n = len(w_market)
    
    # Reverse engineer equilibrium returns
    var_market = w_market @ Sigma @ w_market
    market_premium = 6.0  # Historical equity risk premium (%)
    lambda_risk = market_premium / var_market
    Pi = lambda_risk * Sigma @ w_market
    
    if P is not None and Q is not None:
        # Incorporate views
        Omega = tau * P @ Sigma @ P.T
        
        tau_Sigma_inv = np.linalg.inv(tau * Sigma)
        Omega_inv = np.linalg.inv(Omega)
        
        M_inv = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
        mu_BL = M_inv @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)
        Sigma_BL = Sigma + M_inv
        
        return mu_BL, Sigma_BL, Pi
    
    # Return equilibrium returns even without views
    return Pi, Sigma, Pi

def optimize_portfolio(mu, Sigma, rf, allow_short=False):
    """Optimize portfolio for maximum Sharpe ratio"""
    n = len(mu)
    
    # FIXED: Convert rf from percentage to decimal for calculation
    rf_decimal = rf / 100
    
    # Objective: minimize negative Sharpe ratio
    def neg_sharpe(w):
        port_return = w @ mu / 100  # Convert to decimal
        port_vol = np.sqrt(w @ Sigma @ w) / 100  # Convert to decimal
        if port_vol == 0:
            return 1e10
        return -(port_return - rf_decimal) / port_vol
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Bounds
    if allow_short:
        bounds = tuple((-1, 1) for _ in range(n))
    else:
        bounds = tuple((0, 1) for _ in range(n))
    
    # Initial guess (equal weight)
    w0 = np.array([1/n] * n)
    
    # Optimize
    result = minimize(neg_sharpe, w0, method='SLSQP', 
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000})
    
    if result.success:
        weights = result.x
        port_return = weights @ mu
        port_vol = np.sqrt(weights @ Sigma @ weights)
        sharpe = (port_return - rf) / port_vol
        
        return {
            'weights': weights,
            'return': port_return,
            'volatility': port_vol,
            'sharpe': sharpe
        }
    else:
        st.error("Optimization failed")
        return None

def allocate_risky_riskfree(optimal_portfolio, rf, risk_aversion):
    """Determine allocation between risky portfolio and risk-free asset"""
    mu_p = optimal_portfolio['return']
    sigma_p = optimal_portfolio['volatility']
    
    # Optimal allocation to risky portfolio
    alpha = (mu_p - rf) / (risk_aversion * sigma_p**2)
    alpha = np.clip(alpha, 0, 1)
    
    final_return = alpha * mu_p + (1 - alpha) * rf
    final_vol = alpha * sigma_p
    
    return {
        'alpha': alpha,
        'final_return': final_return,
        'final_volatility': final_vol,
        'risky_weights': optimal_portfolio['weights']
    }

def calculate_risk_metrics(returns, weights, confidence=0.95):
    """Calculate additional risk metrics (VaR, CVaR, Max Drawdown)"""
    portfolio_returns = returns @ weights
    
    # Value at Risk (95%)
    var = np.percentile(portfolio_returns, (1-confidence)*100)
    
    # Conditional VaR (Expected Shortfall)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    
    # Maximum Drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'var_95': var * 100,  # Convert to percentage
        'cvar_95': cvar * 100,
        'max_drawdown': max_drawdown * 100
    }

def check_rebalancing_needed(current_weights, target_weights, threshold=0.05):
    """Check if portfolio needs rebalancing"""
    deviation = np.abs(current_weights - target_weights)
    max_dev = np.max(deviation)
    return max_dev > threshold, max_dev

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("📊 Advanced Portfolio Optimizer")
st.markdown("*Powered by Modern Portfolio Theory & Black-Litterman Model*")

# Sidebar inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Tickers
    tickers_input = st.text_area(
        "Stock Tickers (one per line)",
        value="SPY\nTLT\nGLD\nVNQ\nEFA",
        height=150
    )
    tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]
    
    st.markdown("---")
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        years_back = st.selectbox("Years of Data", [1, 2, 3, 5, 10], index=3)
    with col2:
        frequency = st.selectbox("Frequency", ["weekly", "monthly"], index=0)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back*365)
    
    st.markdown("---")
    
    # Risk parameters
    rf_rate = st.number_input(
        "Risk-Free Rate (% annual)",
        min_value=0.0, max_value=10.0, value=3.8, step=0.1
    )
    
    risk_aversion = st.slider(
        "Risk Aversion (λ)",
        min_value=1.0, max_value=15.0, value=4.0, step=0.5,
        help="1-2: Aggressive | 4-6: Moderate | 10+: Conservative"
    )
    
    st.markdown("---")
    
    # Expected returns method
    returns_method = st.selectbox(
        "Expected Returns Method",
        ["equilibrium", "historical"],
        index=0,
        help="Equilibrium (B-L) is more stable than historical means"
    )
    
    if returns_method == "equilibrium":
        st.info("✅ Using market-implied equilibrium returns (recommended)")
    else:
        st.warning("⚠️ Historical means are noisy and unstable")
    
    st.markdown("---")
    
    # Black-Litterman views
    use_custom_views = st.checkbox("Add Custom Views", value=False)
    
    if use_custom_views:
        st.markdown("**Custom Views**")
        st.caption("Example: 'SPY TLT 3.0' means SPY will outperform TLT by 3%")
        
        views_text = st.text_area(
            "Views (one per line: TICKER1 TICKER2 EXPECTED_DIFF)",
            value="SPY TLT 3.0",
            height=100
        )
        
    st.markdown("---")
    
    optimize_button = st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)

# Main content
if optimize_button:
    with st.spinner("Downloading data and optimizing..."):
        try:
            # Download data with error handling
            prices = download_data(tickers, start_date, end_date)
            
            if prices is None:
                st.stop()
            
            # Update tickers list to only valid ones
            tickers = list(prices.columns)
            
            # Calculate returns
            returns = calculate_returns(prices, frequency)
            
            # Check minimum data requirements
            min_periods = 52 if frequency == 'weekly' else 12
            if len(returns) < min_periods:
                st.error(f"❌ Insufficient data. Need at least {min_periods} periods of returns.")
                st.stop()
            
            # Annualize
            periods_per_year = 52 if frequency == 'weekly' else 12
            Sigma_sample = returns.cov() * periods_per_year * 10000  # Convert to percentage squared
            
            # Get market cap weights
            with st.spinner("Fetching market cap data..."):
                w_market = get_market_cap_weights(tickers)
            
            # Calculate expected returns using selected method
            mu = calculate_expected_returns(
                returns, 
                Sigma_sample.values / 10000,  # Convert back for calculation
                w_market, 
                rf_rate, 
                method=returns_method
            )
            
            # Apply Black-Litterman with custom views if enabled
            if use_custom_views and views_text.strip():
                # Parse views
                views_lines = [line.strip() for line in views_text.split('\n') if line.strip()]
                P_list = []
                Q_list = []
                
                for view in views_lines:
                    parts = view.split()
                    if len(parts) == 3:
                        ticker1, ticker2, diff = parts
                        if ticker1 in tickers and ticker2 in tickers:
                            p_row = np.zeros(len(tickers))
                            p_row[tickers.index(ticker1)] = 1
                            p_row[tickers.index(ticker2)] = -1
                            P_list.append(p_row)
                            Q_list.append(float(diff))
                
                if P_list:
                    P = np.array(P_list)
                    Q = np.array(Q_list)
                    mu_bl, Sigma_bl, Pi = black_litterman(
                        Sigma_sample.values, w_market, rf_rate, P=P, Q=Q
                    )
                    mu = pd.Series(mu_bl, index=tickers)
                    Sigma = pd.DataFrame(Sigma_bl, index=tickers, columns=tickers)
                    st.success("✅ Black-Litterman model applied with your custom views")
                else:
                    Sigma = Sigma_sample
            else:
                Sigma = Sigma_sample
            
            # Optimize
            opt_result = optimize_portfolio(
                mu.values, Sigma.values, rf_rate, allow_short=False
            )
            
            if opt_result:
                # Allocate between risky and risk-free
                allocation = allocate_risky_riskfree(opt_result, rf_rate, risk_aversion)
                
                # Calculate risk metrics
                risk_metrics = calculate_risk_metrics(returns, opt_result['weights'])
                
                # Display results
                st.success("✅ Optimization Complete!")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Expected Return", f"{allocation['final_return']:.2f}%")
                with col2:
                    st.metric("Expected Volatility", f"{allocation['final_volatility']:.2f}%")
                with col3:
                    sharpe = (allocation['final_return'] - rf_rate) / allocation['final_volatility']
                    st.metric("Sharpe Ratio", f"{sharpe:.3f}")
                with col4:
                    st.metric("Risky Allocation", f"{allocation['alpha']*100:.1f}%")
                
                st.markdown("---")
                
                # Tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Allocation", 
                    "💰 Dollar Amounts", 
                    "📈 Efficient Frontier", 
                    "⚠️ Risk Metrics",
                    "📋 Details"
                ])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Final Portfolio Allocation")
                        
                        # Prepare data for pie chart
                        final_weights = {'Risk-Free (Bank)': (1 - allocation['alpha']) * 100}
                        for i, ticker in enumerate(tickers):
                            weight = allocation['alpha'] * opt_result['weights'][i] * 100
                            if weight > 0.5:  # Only show if > 0.5%
                                final_weights[ticker] = weight
                        
                        fig = px.pie(
                            values=list(final_weights.values()),
                            names=list(final_weights.keys()),
                            title="Portfolio Composition"
                        )
                        fig.update_traces(textposition='inside', textinfo='label+percent')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Risky Portfolio Breakdown")
                        
                        risky_df = pd.DataFrame({
                            'Asset': tickers,
                            'Weight in Risky (%)': opt_result['weights'] * 100,
                            'Weight in Total (%)': allocation['alpha'] * opt_result['weights'] * 100
                        }).sort_values('Weight in Total (%)', ascending=False)
                        
                        fig = px.bar(
                            risky_df,
                            x='Asset',
                            y='Weight in Total (%)',
                            title="Asset Weights in Total Portfolio"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.subheader("💵 Investment Calculator")
                    
                    portfolio_value = st.number_input(
                        "Total Portfolio Value ($)",
                        min_value=1000,
                        max_value=10000000,
                        value=100000,
                        step=1000
                    )
                    
                    # Calculate dollar amounts
                    dollar_allocation = []
                    dollar_allocation.append({
                        'Asset': 'Risk-Free (Bank)',
                        'Weight (%)': (1 - allocation['alpha']) * 100,
                        'Amount ($)': (1 - allocation['alpha']) * portfolio_value
                    })
                    
                    for i, ticker in enumerate(tickers):
                        weight = allocation['alpha'] * opt_result['weights'][i]
                        dollar_allocation.append({
                            'Asset': ticker,
                            'Weight (%)': weight * 100,
                            'Amount ($)': weight * portfolio_value
                        })
                    
                    dollar_df = pd.DataFrame(dollar_allocation)
                    dollar_df = dollar_df.sort_values('Amount ($)', ascending=False)
                    
                    # Display table
                    st.dataframe(
                        dollar_df.style.format({
                            'Weight (%)': '{:.2f}',
                            'Amount ($)': '${:,.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Bar chart
                    fig = px.bar(
                        dollar_df,
                        x='Asset',
                        y='Amount ($)',
                        title=f"Portfolio Allocation for ${portfolio_value:,.0f}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    csv = dollar_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Allocation (CSV)",
                        data=csv,
                        file_name=f"portfolio_allocation_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with tab3:
                    st.subheader("Efficient Frontier")
                    
                    # Generate random portfolios
                    n_portfolios = 5000
                    results = np.zeros((3, n_portfolios))
                    
                    np.random.seed(42)
                    for i in range(n_portfolios):
                        weights = np.random.random(len(tickers))
                        weights /= weights.sum()
                        
                        results[0, i] = weights @ mu.values
                        results[1, i] = np.sqrt(weights @ Sigma.values @ weights)
                        results[2, i] = (results[0, i] - rf_rate) / results[1, i]
                    
                    # Create scatter plot
                    fig = go.Figure()
                    
                    # Random portfolios
                    fig.add_trace(go.Scatter(
                        x=results[1, :],
                        y=results[0, :],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=results[2, :],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Sharpe<br>Ratio")
                        ),
                        name='Random Portfolios'
                    ))
                    
                    # Risk-free asset
                    fig.add_trace(go.Scatter(
                        x=[0],
                        y=[rf_rate],
                        mode='markers',
                        marker=dict(size=15, color='blue'),
                        name='Risk-Free Asset'
                    ))
                    
                    # Optimal risky portfolio
                    fig.add_trace(go.Scatter(
                        x=[opt_result['volatility']],
                        y=[opt_result['return']],
                        mode='markers',
                        marker=dict(size=15, color='red'),
                        name='Optimal Risky Portfolio'
                    ))
                    
                    # Your final portfolio
                    fig.add_trace(go.Scatter(
                        x=[allocation['final_volatility']],
                        y=[allocation['final_return']],
                        mode='markers',
                        marker=dict(size=15, color='green'),
                        name='Your Portfolio'
                    ))
                    
                    # Capital allocation line
                    x_line = np.linspace(0, opt_result['volatility'] * 1.2, 100)
                    sharpe_line = (opt_result['return'] - rf_rate) / opt_result['volatility']
                    y_line = rf_rate + sharpe_line * x_line
                    
                    fig.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        line=dict(color='darkgreen', width=2, dash='dash'),
                        name='Capital Allocation Line'
                    ))
                    
                    fig.update_layout(
                        title="Efficient Frontier",
                        xaxis_title="Volatility (%)",
                        yaxis_title="Expected Return (%)",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("""
                    **How to read this chart:**
                    - **Gray dots**: All possible portfolio combinations
                    - **Blue dot**: Risk-free asset (bank account)
                    - **Red dot**: Optimal risky portfolio (maximum Sharpe ratio)
                    - **Green dot**: Your final portfolio (mix of risky + risk-free)
                    - **Dashed line**: Capital allocation line (best risk/return tradeoff)
                    """)
                
                with tab4:
                    st.subheader("⚠️ Risk Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Value at Risk (95%)", 
                            f"{risk_metrics['var_95']:.2f}%",
                            help="Maximum expected loss in 95% of scenarios"
                        )
                    
                    with col2:
                        st.metric(
                            "Conditional VaR (95%)", 
                            f"{risk_metrics['cvar_95']:.2f}%",
                            help="Expected loss in worst 5% of scenarios"
                        )
                    
                    with col3:
                        st.metric(
                            "Max Drawdown", 
                            f"{risk_metrics['max_drawdown']:.2f}%",
                            help="Largest peak-to-trough decline in historical data"
                        )
                    
                    st.markdown("---")
                    
                    # Rebalancing monitor
                    st.subheader("📊 Rebalancing Monitor")
                    
                    threshold = st.slider(
                        "Rebalancing Threshold (%)", 
                        1, 10, 5,
                        help="Trigger rebalancing when any asset deviates by this percentage"
                    ) / 100
                    
                    st.info(f"""
                    **Recommended Rebalancing Strategy:**
                    - Review portfolio **monthly**
                    - Rebalance if any asset deviates by more than **{threshold*100:.0f}%**
                    - This minimizes transaction costs while maintaining target allocation
                    """)
                    
                    st.markdown("---")
                    
                    # Risk interpretation
                    st.subheader("💡 Risk Metric Interpretation")
                    
                    st.markdown(f"""
                    **For your portfolio:**
                    
                    1. **Value at Risk (VaR)**: With 95% confidence, your portfolio won't lose more than 
                       **{abs(risk_metrics['var_95']):.2f}%** in a single period.
                    
                    2. **Conditional VaR**: In the worst 5% of scenarios, you can expect to lose 
                       **{abs(risk_metrics['cvar_95']):.2f}%** on average.
                    
                    3. **Maximum Drawdown**: Historically, the largest decline from peak was 
                       **{abs(risk_metrics['max_drawdown']):.2f}%**.
                    
                    These metrics help you understand downside risk beyond just volatility.
                    """)
                
                with tab5:
                    st.subheader("Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Expected Returns (Annual %)**")
                        returns_df = pd.DataFrame({
                            'Asset': mu.index,
                            'Expected Return (%)': mu.values
                        }).sort_values('Expected Return (%)', ascending=False)
                        st.dataframe(returns_df, use_container_width=True)
                        
                        if returns_method == 'equilibrium':
                            st.caption("✅ Using equilibrium (market-implied) returns")
                        else:
                            st.caption("⚠️ Using historical average returns")
                    
                    with col2:
                        st.markdown("**Volatility (Annual %)**")
                        vol_df = pd.DataFrame({
                            'Asset': tickers,
                            'Volatility (%)': np.sqrt(np.diag(Sigma.values))
                        }).sort_values('Volatility (%)', ascending=False)
                        st.dataframe(vol_df, use_container_width=True)
                    
                    st.markdown("**Correlation Matrix**")
                    corr = returns.corr()
                    fig = px.imshow(
                        corr,
                        text_auto='.2f',
                        aspect='auto',
                        color_continuous_scale='RdBu_r',
                        title="Asset Correlation Matrix",
                        zmin=-1, zmax=1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("**Market Capitalization Weights**")
                    market_df = pd.DataFrame({
                        'Asset': tickers,
                        'Market Weight (%)': w_market * 100
                    }).sort_values('Market Weight (%)', ascending=False)
                    st.dataframe(market_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check your ticker symbols and try again.")
            import traceback
            with st.expander("Show detailed error"):
                st.code(traceback.format_exc())

else:
    # Landing page
    st.info("👈 Configure your portfolio settings in the sidebar and click 'Optimize Portfolio'")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - Modern Portfolio Theory
        - Black-Litterman Model
        - **Equilibrium Returns (Default)**
        - Risk-Free Asset Allocation
        - **VaR & CVaR Analysis**
        - **Maximum Drawdown**
        - Interactive Visualizations
        - Dollar Amount Calculator
        """)
    
    with col2:
        st.markdown("### 💡 How It Works")
        st.markdown("""
        1. Enter stock tickers
        2. Set risk parameters
        3. Choose returns method
        4. Add custom views (optional)
        5. Get optimal allocation
        6. See detailed risk metrics
        7. Calculate dollar amounts
        """)
    
    with col3:
        st.markdown("### 📊 Risk Aversion Guide")
        st.markdown("""
        - **λ = 1-2**: Very Aggressive
        - **λ = 2-4**: Aggressive  
        - **λ = 4-6**: Moderate
        - **λ = 6-10**: Conservative
        - **λ = 10+**: Very Conservative
        
        *Higher λ = More allocation to risk-free asset*
        """)
    
    st.markdown("---")
    
    # Key improvements section
    st.markdown("### 🚀 Key Improvements")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Fixed Issues:**
        - Risk-free rate conversion bug
        - Consistent return calculations
        - Market-cap weighted equilibrium
        - Robust error handling
        - Minimum data validation
        """)
    
    with col2:
        st.markdown("""
        **✨ New Features:**
        - Equilibrium returns (default)
        - Value at Risk (VaR)
        - Conditional VaR (CVaR)
        - Maximum Drawdown
        - Rebalancing monitor
        """)

# Footer
st.markdown("---")
st.caption("Data provided by Yahoo Finance. Past performance does not guarantee future results. This tool is for educational purposes only. Not investment advice.")
