# app.py
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
# HELPER FUNCTIONS
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
    insufficient_data_tickers = [ticker for ticker in tickers if data[ticker].count() < 52] # Min 1 year of weekly data
    if insufficient_data_tickers:
        st.warning(f"⚠️ Insufficient data for: {', '.join(insufficient_data_tickers)}. They will be excluded.")
        data.drop(columns=insufficient_data_tickers, inplace=True)
        
    return data

@st.cache_data(ttl=36000) # Cache for 10 hours
def get_market_cap_weights(tickers):
    """Get market capitalization weights with robust error handling"""
    market_caps = {}
    failed_tickers = []

    with st.spinner("Fetching market capitalization / asset-size data..."):
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info

                # Heuristic to detect ETFs vs equities
                # Prefer 'totalAssets' for ETFs (many funds expose this key)
                is_etf = False
                if 'totalAssets' in info and info.get('totalAssets') is not None:
                    is_etf = True
                elif str(info.get('quoteType', '')).upper() == 'ETF':
                    is_etf = True
                elif info.get('fundFamily') is not None or info.get('category') is not None or info.get('netAssets') is not None:
                    is_etf = True

                # Choose the appropriate size metric
                if is_etf:
                    mc = info.get('totalAssets') or info.get('netAssets') or info.get('marketCap')
                else:
                    mc = info.get('marketCap') or info.get('totalAssets')

                # Accept only positive numeric sizes
                if mc and isinstance(mc, (int, float)) and mc > 0:
                    market_caps[ticker] = mc
                else:
                    failed_tickers.append(ticker)
            except Exception:
                failed_tickers.append(ticker)
    # Return raw fetched sizes and list of failed tickers for UI handling
    return market_caps, failed_tickers

def black_litterman(Sigma_percent_sq, w_market, rf_percent, market_risk_premium, tau=0.025, P=None, Q_percent=None):
    """
    Black-Litterman model implementation.
    This function now correctly handles all unit conversions internally.
    """
    n = len(w_market)
    rf_decimal = rf_percent / 100
    Sigma_decimal = Sigma_percent_sq / 10000  # Convert from %-squared to decimal
    market_premium_decimal = market_risk_premium / 100  # Convert from % to decimal

    # 1. Reverse engineer equilibrium returns (in decimal)
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


def calculate_returns(prices, frequency='weekly'):
    """Calculate arithmetic returns at specified frequency"""
    if frequency == 'weekly':
        resample_rule = 'W-FRI'  # Use Friday to ensure consistency
        prices_resampled = prices.resample(resample_rule).last()
    elif frequency == 'monthly':
        resample_rule = 'M'
        prices_resampled = prices.resample(resample_rule).last()
    else:  # Daily
        prices_resampled = prices

    returns = prices_resampled.pct_change().dropna()
    return returns


def calculate_expected_returns(returns, Sigma_decimal, w_market, rf_rate_percent, market_risk_premium, method='equilibrium'):
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
        market_risk_premium_decimal = market_risk_premium / 100
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

def calculate_gap_views(returns, Sigma_decimal, w_market, rf_rate, market_risk_premium,
                       confidence=0.3, min_gap_threshold=1.0):
    """
    Calculate Black-Litterman views based on the gap between historical and equilibrium returns.
    
    This integrates historical momentum into equilibrium returns in a statistically principled way.
    
    Parameters:
    - confidence: How much of the historical outperformance to incorporate (0-1)
                 0.3 = "30% of past momentum will persist" (moderate/conservative)
    - min_gap_threshold: Only create views for assets with gaps larger than this (in %)
    
    Returns: P matrix, Q vector, and gaps DataFrame for display
    """
    periods_per_year = 52 if st.session_state.frequency == 'weekly' else 12
    
    # Calculate both methods
    mu_hist = calculate_expected_returns(returns, Sigma_decimal, w_market, 
                                         rf_rate, market_risk_premium, method='historical')
    mu_eq = calculate_expected_returns(returns, Sigma_decimal, w_market, 
                                       rf_rate, market_risk_premium, method='equilibrium')
    
    # Calculate gaps (positive = historical > equilibrium, indicating momentum)
    gaps = mu_hist - mu_eq
    
    # Sort by absolute gap magnitude
    sorted_gaps = gaps.abs().sort_values(ascending=False)
    
    P_list, Q_list = [], []
    
    # Create views for assets with significant gaps
    for ticker in sorted_gaps.index:
        gap = gaps[ticker]
        
        if abs(gap) > min_gap_threshold:
            # View: This asset will outperform/underperform the market-weighted portfolio
            p_row = np.zeros(len(returns.columns))
            p_row[returns.columns.tolist().index(ticker)] = 1
            
            # Subtract market weights from all other assets (creates view vs. market portfolio)
            w_others = w_market.copy()
            w_others[returns.columns.tolist().index(ticker)] = 0
            if w_others.sum() > 0:  # Avoid division by zero
                w_others = w_others / w_others.sum()  # Renormalize
            
            for i, t in enumerate(returns.columns):
                if t != ticker:
                    p_row[i] = -w_others[i]
            
            # View magnitude: confidence × gap
            # E.g., if gap = 10% and confidence = 0.3, view = 3%
            view_magnitude = confidence * gap
            
            P_list.append(p_row)
            Q_list.append(view_magnitude)
    
    if len(P_list) == 0:
        return None, None, gaps
    
    return np.array(P_list), np.array(Q_list), gaps


# ----------------------------- Portfolio Comparison Utils -----------------------------
def parse_actual_portfolio(text):
    """Parse user-entered portfolio text in the form 'TICKER WEIGHT' per line.
    We accept weights as decimals (0.25) or percents (25 or 25%). Returns dict ticker->weight (raw),
    then normalized to sum to 1.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    parsed = {}
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        t = parts[0].upper()
        val = parts[1].replace('%', '')
        try:
            v = float(val)
        except Exception:
            continue
        # If user likely entered percents (>1 and not a decimal), treat as percent
        if v > 1 and v <= 100:
            v = v / 100.0
        parsed[t] = v
    # Normalize
    total = sum(parsed.values())
    if total <= 0:
        return {}
    for k in parsed:
        parsed[k] = parsed[k] / total
    return parsed


def align_weights(universe, weights_dict):
    """Return numpy vector of weights aligned to 'universe' (list of tickers) given a dict.
    Missing tickers become 0.
    """
    return np.array([weights_dict.get(t, 0.0) for t in universe])


def portfolio_stats_from_weights(weights, mu_percent, Sigma_percent_sq, rf_percent):
    """Compute portfolio return, volatility, and sharpe from weights.
    mu_percent: array-like of annual expected returns in percent
    Sigma_percent_sq: covariance matrix in %-squared
    Returns dict with return(%), vol(%), sharpe
    """
    w = np.array(weights)
    port_return = float(w @ np.array(mu_percent))
    port_vol = float(np.sqrt(w @ np.array(Sigma_percent_sq) @ w))
    sharpe = (port_return - rf_percent) / port_vol if port_vol > 0 else 0.0
    return {'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}


def tracking_error_and_mahalanobis(w_actual, w_optimal, Sigma_decimal):
    """Compute tracking error (percent) and Mahalanobis distance using decimal covariance."""
    d = w_actual - w_optimal
    # Tracking error (decimal) = sqrt(d^T Sigma_decimal d)
    te_decimal = np.sqrt(float(d @ Sigma_decimal @ d)) if d.size > 0 else 0.0
    te_percent = te_decimal * 100

    # Mahalanobis using pseudo-inverse for stability
    try:
        Sigma_inv = np.linalg.pinv(Sigma_decimal)
        m = np.sqrt(float(d @ Sigma_inv @ d))
    except Exception:
        m = float(np.sqrt(d @ d))

    return te_percent, m


def substitution_analysis(actual_only, returns_df, mu_series, opt_portfolio_returns):
    """For assets present in actual_only (list), compute correlation with optimal portfolio,
    return difference, and quality score per guide.
    returns_df: aligned returns DataFrame columns are tickers
    mu_series: expected returns (annual %) indexed by ticker
    opt_portfolio_returns: series of portfolio returns per period for the optimal portfolio
    """
    rows = []
    for t in actual_only:
        if t not in returns_df.columns:
            continue
        series = returns_df[t]
        corr = series.corr(opt_portfolio_returns)
        asset_return = float(mu_series.get(t, np.nan))
        opt_return = float(mu_series.mean()) if len(mu_series) > 0 else np.nan
        ret_diff = asset_return - opt_return
        quality = corr - abs(ret_diff) / 10.0
        rows.append({'Ticker': t, 'Correlation': corr, 'Return (%)': asset_return, 'Return vs Optimal (%)': ret_diff, 'Quality': quality})
    return pd.DataFrame(rows).set_index('Ticker') if rows else pd.DataFrame()


def overall_deviation_score(sharpe_gap, tracking_error_percent, return_gap_percent, vol_gap_percent):
    """Compute composite deviation score as described in the guide."""
    sharpe_penalty = abs(sharpe_gap) * 30
    tracking_penalty = tracking_error_percent * 2
    return_penalty = abs(return_gap_percent) * 1.5
    vol_penalty = abs(vol_gap_percent) * 1.0
    total = sharpe_penalty + tracking_penalty + return_penalty + vol_penalty
    return min(100.0, total)


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
    risk_aversion = st.slider("Risk Aversion (RA)", 1.0, 15.0, 5.0, 0.5, help="Higher RA means more conservative. 2-4: Aggressive, 4-6: Moderate, 6-10: Conservative, 10+: Very Conservative")
    st.markdown("---")
    returns_method = st.selectbox("Expected Returns Method", ["equilibrium", "historical"], index=0,
        help="Equilibrium: Market-implied returns based on current prices and risk. Historical: Simple average of past returns.")
    
    if returns_method == "equilibrium": 
        st.info("✅ **Using Equilibrium Returns**")
        with st.expander("ℹ️ What does this mean?"):
            st.markdown("""
            **Equilibrium returns** represent market consensus expectations:
            - Derived from current market prices and asset volatilities
            - Assumes the market is in equilibrium (supply = demand)
            - Uses reverse optimization from CAPM theory
            - Formula: Π = λ × Σ × w_market (where λ = market risk aversion)
            - More stable and forward-looking than historical averages
            - Not biased by recent performance anomalies
                        
            **Advantage**: Theoretically grounded, less prone to estimation error
            """)
    else: 
        st.warning("⚠️ **Using Historical Average Returns**")
        with st.expander("ℹ️ Limitations of historical returns"):
            st.markdown("""
            **Historical means** extrapolate past performance into the future:
            - **High estimation error**: Standard error often exceeds the mean itself
            - **Time period sensitivity**: Results change dramatically with different lookback periods
            - **No mean reversion**: Assumes past outperformers will continue outperforming
            - **Recency bias**: Overly influenced by recent trends
            - Can lead to extreme, concentrated portfolios
            
            **Academic consensus**: Historical means are poor predictors of future returns (see Merton 1980, Michaud 1989).
            """)
    
    st.markdown("---")
    
    # Historical Momentum Integration (Only available for equilibrium method)
    if returns_method == "equilibrium":
        use_momentum_views = st.checkbox("📈 Historical Momentum Integration", value=True,
            help="Blend recent performance trends with equilibrium returns using Black-Litterman model")
        
        if use_momentum_views:
            st.markdown("**Black-Litterman Settings**")
            momentum_confidence = st.slider(
                "Momentum Persistence (%)", 
                0, 100, 30, 5,
                help="How much of the historical outperformance do you expect to continue? 30% = moderate, 50% = balanced, 70% = aggressive"
            ) / 100
            
            min_gap = st.slider(
                "Minimum Significance (%)", 
                0.5, 5.0, 1.0, 0.5,
                help="Only incorporate momentum for assets that outperformed/underperformed by at least this amount"
            )
            
            with st.expander("ℹ️ How Momentum Integration Works"):
                st.markdown("""
                This feature uses the **Black-Litterman model** to blend equilibrium returns with recent trends:
                
                1. **Calculate the gap**: Historical return - Equilibrium return
                2. **Scale by confidence**: If Gold's gap is +10% and confidence is 30%, the view is +3%
                3. **Bayesian blending**: The model statistically combines equilibrium with your momentum views
                
                **Example**: 
                - Gold historical return: 15%
                - Gold equilibrium return: 5%
                - Gap: 10%
                - With 30% confidence → Black-Litterman adjusts Gold's expected return upward by ~3%
                
                **Why this works**:
                - Momentum persists in the medium term (6-12 months) - empirical fact
                - But it's not fully predictive - mean reversion also exists
                - This approach balances both effects in a statistically principled way
                """)
    else:
        # Historical method is selected - momentum integration is not applicable
        use_momentum_views = False
        st.info("ℹ️ **Momentum Integration**: Not applicable when using Historical returns (already incorporates past performance)")
    
    if 'use_momentum_views' not in locals():
        use_momentum_views = False
    if 'momentum_confidence' not in locals():
        momentum_confidence = 0.3
    if 'min_gap' not in locals():
        min_gap = 1.0
    
    st.markdown("---")
    
    # Manual custom views (optional, for advanced users)
    use_custom_views = st.checkbox("Advanced: Add Manual Black-Litterman Views",
        help="Specify your own views about relative performance (e.g., 'GLD SPY 2.5' means Gold will outperform SPY by 2.5%)")
    
    # Persist views in session state so they survive reruns
    if 'views_text' not in st.session_state:
        st.session_state['views_text'] = "GLD SPY 0.5\nGLD VTI 0.5"

    if use_custom_views:
        st.markdown("**Your Views (one per line: TICKER1 TICKER2 VALUE)**")
        views_text = st.text_area("Format: ASSET1 ASSET2 OUTPERFORMANCE%", 
            value=st.session_state['views_text'], height=100, key='views_text',
            help="Example: 'GLD SPY 2.5' means you believe Gold will outperform SPY by 2.5% annually")
        
        with st.expander("📖 View Format Examples"):
            st.markdown("""
            - `GLD SPY 3.0` → Gold outperforms SPY by 3%
            - `SPY CIBR -1.5` → SPY underperforms CIBR by 1.5% (or CIBR beats SPY by 1.5%)
            - One view per line
            - Value is the expected annual return difference in percentage points
            """)
    else:
        views_text = st.session_state.get('views_text', "GLD SPY 0.5\nGLD VTI 0.5")
    
    st.markdown("---")
    
    # Market Risk Premium (Advanced setting, collapsed by default)
    with st.expander("🔧 Advanced: Market Risk Premium"):
        market_risk_premium = st.slider(
            "Assumed Market Risk Premium (Annual %)", 
            5.0, 15.0, 10.0, 0.5,
            help="Historical equity risk premium over risk-free rate. Used in equilibrium return calculations. Default: 10% (historical US equity premium)"
        )
        st.caption("💡 This affects equilibrium return calculations. 10% is the long-term historical equity risk premium in the US.")
    
    if 'market_risk_premium' not in locals():
        market_risk_premium = 10.0  # Default if expander not opened
    
    st.markdown("---")
    optimize_button = st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)
    # Allow results to persist across reruns
    if 'show_results' not in st.session_state:
        st.session_state['show_results'] = False

run_opt = optimize_button or st.session_state.get('show_results', False)

if run_opt:
    with st.spinner("Processing... This may take a moment."):
        try:
            # If this execution is a rerun triggered by a later button (e.g., Compare Portfolios),
            # restore data from session_state; otherwise perform a fresh run.
            if not optimize_button and st.session_state.get('show_results'):
                prices = st.session_state.get('cmp_prices')
                tickers = st.session_state.get('cmp_tickers')
                returns = st.session_state.get('cmp_returns')
                Sigma_percent_sq = st.session_state.get('cmp_Sigma_percent_sq')
                # If Sigma_percent_sq was stored as a numpy array, convert it back to DataFrame
                if Sigma_percent_sq is not None and not hasattr(Sigma_percent_sq, 'columns'):
                    Sigma_percent_sq = pd.DataFrame(np.array(Sigma_percent_sq), index=tickers, columns=tickers)
                mu = st.session_state.get('cmp_mu')
                Sigma_decimal = Sigma_percent_sq.values / 10000 if Sigma_percent_sq is not None else None
                w_market = st.session_state.get('cmp_w_market')
                market_caps = st.session_state.get('cmp_market_caps', {})
                failed_tickers = st.session_state.get('cmp_failed_tickers', [])
                opt_result = st.session_state.get('cmp_opt_result')
            else:
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

            with st.spinner("Fetching market capitalization / asset-size data..."):
                market_caps, failed_tickers = get_market_cap_weights(tickers)

            # Provide manual override inputs in the sidebar for any failed tickers
            overrides = {}
            if failed_tickers:
                # Initialize session storage for overrides if not present
                if 'overrides' not in st.session_state:
                    st.session_state['overrides'] = {}

                with st.sidebar.expander("Manual size overrides for missing tickers", expanded=True):
                    st.markdown("If any tickers failed to fetch sizes, enter manual sizes below. Leave 0 to skip.")
                    unit = st.selectbox("Unit for manual overrides", ("Billion", "Million"), index=0)
                    mult = 1e9 if unit == "Billion" else 1e6
                    for t in failed_tickers:
                        key = f"override_{t}"
                        # Prefill from session state if present
                        pre = st.session_state['overrides'].get(t, 0.0) / mult if st.session_state['overrides'].get(t, 0) else 0.0
                        val = st.number_input(f"{t} size ({unit})", min_value=0.0, value=float(pre), step=0.1, format="%.3f", key=key)
                        if val and val > 0:
                            overrides[t] = float(val) * mult
                            # Persist to session state
                            st.session_state['overrides'][t] = float(val) * mult


            # Merge fetched sizes with overrides (overrides take precedence)
            merged_sizes = dict(market_caps) if market_caps else {}
            for t, v in overrides.items():
                merged_sizes[t] = v

            # Determine missing tickers after overrides
            missing_after = [t for t in tickers if t not in merged_sizes]

            # Build a display table with source info
            def fmt_size(v):
                if v is None:
                    return "N/A"
                if abs(v) >= 1e9:
                    return f"${v/1e9:.2f}B"
                if abs(v) >= 1e6:
                    return f"${v/1e6:.2f}M"
                return f"${v:.0f}"

            display_rows = []
            for t in tickers:
                if t in merged_sizes:
                    src = 'Overridden' if t in overrides else 'Fetched'
                    display_rows.append({'Ticker': t, 'Asset Size': fmt_size(merged_sizes[t]), 'Source': src})
                else:
                    display_rows.append({'Ticker': t, 'Asset Size': 'N/A', 'Source': 'Missing'})

            with st.expander("📊 View Asset-Size Data & Weights", expanded=False):
                df_display = pd.DataFrame(display_rows).set_index('Ticker')
                # Highlight missing rows
                def highlight_missing(s):
                    return ['background-color: #ffcccc' if v == 'Missing' else '' for v in s]

                st.dataframe(df_display.style.apply(highlight_missing, subset=['Source']), use_container_width=True)

            # Compute final market weights if we have sizes for all tickers; otherwise fallback to equal weights
            if len(missing_after) == 0 and len(merged_sizes) > 0:
                total_cap = sum(merged_sizes[t] for t in tickers)
                if total_cap <= 0:
                    st.error("Fetched sizes sum to zero or negative. Falling back to equal weights.")
                    w_market = np.array([1/len(tickers)] * len(tickers))
                else:
                    w_market = np.array([merged_sizes[t] / total_cap for t in tickers])
                    st.success("✅ Using fetched and/or overridden sizes to compute market weights.")
            else:
                if missing_after:
                    st.warning(f"⚠️ Size data missing for: {', '.join(missing_after)}. Using equal weights as fallback.")
                    st.info("You can enter overrides for missing tickers in the sidebar expander titled 'Manual size overrides for missing tickers'.")
                w_market = np.array([1/len(tickers)] * len(tickers))

            mu = calculate_expected_returns(returns, Sigma_decimal, w_market, rf_rate, market_risk_premium, returns_method)
            
            # Apply Historical Momentum Integration (only when using equilibrium method)
            if returns_method == "equilibrium" and use_momentum_views:
                P_momentum, Q_momentum, gaps = calculate_gap_views(
                    returns, Sigma_decimal, w_market, rf_rate, market_risk_premium,
                    confidence=momentum_confidence, min_gap_threshold=min_gap
                )
                
                if P_momentum is not None and len(P_momentum) > 0:
                    # Apply Black-Litterman with momentum views
                    mu_bl, Sigma_bl, pi_eq = black_litterman(
                        Sigma_percent_sq.values, w_market, rf_rate, market_risk_premium,
                        P=P_momentum, Q_percent=Q_momentum
                    )
                    mu = pd.Series(mu_bl, index=tickers)
                    Sigma_percent_sq = pd.DataFrame(Sigma_bl, index=tickers, columns=tickers)
                    
                    # Show what momentum integration did
                    with st.expander("📈 Momentum Integration Results"):
                        st.success(f"✅ Generated {len(Q_momentum)} momentum-based views")
                        
                        # Create comparison table
                        mu_base = calculate_expected_returns(returns, Sigma_decimal, w_market, 
                                                             rf_rate, market_risk_premium, returns_method)
                        momentum_df = pd.DataFrame({
                            'Historical': calculate_expected_returns(returns, Sigma_decimal, w_market,
                                                                    rf_rate, market_risk_premium, 'historical'),
                            'Equilibrium': mu_base,
                            'Gap': gaps,
                            'Momentum-Adjusted': mu
                        }, index=tickers)
                        
                        st.dataframe(
                            momentum_df.style.format({
                                'Historical': '{:.2f}%',
                                'Equilibrium': '{:.2f}%',
                                'Gap': '{:.2f}%',
                                'Momentum-Adjusted': '{:.2f}%'
                            }).background_gradient(subset=['Gap'], cmap='RdYlGn', vmin=-10, vmax=10),
                            use_container_width=True
                        )
                        
                        st.caption(f"💡 Momentum confidence: {momentum_confidence*100:.0f}% | " +
                                 f"Minimum significance: {min_gap:.1f}%")
                else:
                    st.info("ℹ️ No significant momentum detected (all gaps below threshold)")
            
            if use_custom_views and views_text.strip():
                views_lines = [line.strip() for line in views_text.split('\n') if line.strip()]
                P_list, Q_list = [], []
                for view in views_lines:
                    parts = view.split()
                    # Expect exactly: ASSET1 ASSET2 VALUE
                    if len(parts) != 3:
                        st.warning(f"Skipping malformed view (expect 3 fields): '{view}'")
                        continue

                    a, b, val = parts[0].upper(), parts[1].upper(), parts[2]

                    # Reject views where both sides are '*'
                    if a == '*' and b == '*':
                        st.warning(f"Skipping view with both sides '*' (ambiguous): '{view}'")
                        continue

                    # Try to parse numeric value
                    try:
                        q_value = float(val)
                    except Exception:
                        st.warning(f"Skipping view with invalid numeric value: '{view}'")
                        continue

                    # Helper to append a single view given two valid tickers
                    def append_view(t1, t2, q):
                        if t1 not in tickers or t2 not in tickers:
                            return
                        if t1 == t2:
                            # skip self-referential views
                            return
                        p_row = np.zeros(len(tickers))
                        p_row[tickers.index(t1)] = 1
                        p_row[tickers.index(t2)] = -1
                        P_list.append(p_row)
                        Q_list.append(q)

                    # Expand when one side is '*'
                    if a == '*':
                        # a = every ticker except b
                        if b not in tickers:
                            st.warning(f"Ticker '{b}' in view not in current asset list: '{view}'")
                            continue
                        for t in tickers:
                            if t == b:
                                continue
                            append_view(t, b, q_value)
                        continue

                    if b == '*':
                        # b = every ticker except a
                        if a not in tickers:
                            st.warning(f"Ticker '{a}' in view not in current asset list: '{view}'")
                            continue
                        for t in tickers:
                            if t == a:
                                continue
                            append_view(a, t, q_value)
                        continue

                    # No wildcard: both must be valid tickers
                    if a in tickers and b in tickers:
                        append_view(a, b, q_value)
                    else:
                        missing = [x for x in (a, b) if x not in tickers]
                        st.warning(f"Skipping view because these tickers are not in the asset list: {', '.join(missing)}")
                if P_list:
                    P, Q = np.array(P_list), np.array(Q_list)
                    mu_bl, Sigma_bl, pi_eq = black_litterman(Sigma_percent_sq.values, w_market, rf_rate, market_risk_premium, P=P, Q_percent=Q)
                    mu = pd.Series(mu_bl, index=tickers)
                    Sigma_percent_sq = pd.DataFrame(Sigma_bl, index=tickers, columns=tickers)
                    st.success("✅ Black-Litterman model applied with your custom views.")

            opt_result = optimize_portfolio(mu.values, Sigma_percent_sq.values, rf_rate)
            
            if opt_result:
                allocation = allocate_risky_riskfree(opt_result, rf_rate, risk_aversion)
                risk_metrics = calculate_risk_metrics(returns, opt_result['weights'])
                st.success("✅ Optimization Complete!")
                # Persist relevant objects so follow-up actions (like comparisons) work across reruns
                st.session_state['show_results'] = True
                st.session_state['cmp_prices'] = prices
                st.session_state['cmp_tickers'] = tickers
                st.session_state['cmp_returns'] = returns
                # store Sigma_percent_sq as plain array or DataFrame (picklable)
                st.session_state['cmp_Sigma_percent_sq'] = Sigma_percent_sq.values if hasattr(Sigma_percent_sq, 'values') else Sigma_percent_sq
                st.session_state['cmp_mu'] = mu
                st.session_state['cmp_opt_result'] = opt_result
                st.session_state['cmp_w_market'] = w_market
                st.session_state['cmp_market_caps'] = market_caps
                st.session_state['cmp_failed_tickers'] = failed_tickers
                
                # --- DIAGNOSTIC INFO ---
                with st.expander("🔍 Optimization Process Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Step 1: Optimal Risky Portfolio (Tangency)**")
                        st.write(f"Expected Return: {opt_result['return']:.2f}%")
                        st.write(f"Volatility: {opt_result['volatility']:.2f}%")
                        st.write(f"Sharpe Ratio: {opt_result['sharpe']:.3f}")
                    with col2:
                        st.markdown("**Step 2: Risk-Free Asset Allocation**")
                        st.write(f"Risk Aversion (RA): {risk_aversion}")
                        st.write(f"Optimal Risky %: {allocation['alpha']*100:.1f}%")
                        st.write(f"Risk-Free %: {(1-allocation['alpha'])*100:.1f}%")
                    
                    st.markdown("**Formula Used**: α* = (E[R_p] - R_f) / (RA × σ²_p)")
                    
                    # Show the calculation
                    alpha_calc = (opt_result['return']/100 - rf_rate/100) / (risk_aversion * (opt_result['volatility']/100)**2)
                    st.write(f"Calculated α* = ({opt_result['return']:.2f}% - {rf_rate}%) / ({risk_aversion} × {opt_result['volatility']:.2f}%²)")
                    st.write(f"= {opt_result['return']/100 - rf_rate/100:.4f} / {risk_aversion * (opt_result['volatility']/100)**2:.6f} = {alpha_calc:.3f}")
                    if alpha_calc > 1:
                        st.info(f"📊 Calculated α* = {alpha_calc:.3f} was clipped to 1.0 (no leverage constraint)")
                    elif alpha_calc < 0:
                        st.info(f"📊 Calculated α* = {alpha_calc:.3f} was clipped to 0.0 (no shorting constraint)")
                
                # --- RESULTS ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Final Return", f"{allocation['final_return']:.2f}%")
                col2.metric("Final Volatility", f"{allocation['final_volatility']:.2f}%")
                sharpe = (allocation['final_return'] - rf_rate) / allocation['final_volatility'] if allocation['final_volatility'] > 0 else 0
                col3.metric("Sharpe Ratio", f"{sharpe:.3f}")
                col4.metric("Risky Allocation", f"{allocation['alpha']*100:.1f}%")
                st.markdown("---")
                
                # Use a persistent selector instead of st.tabs so the active view doesn't jump on reruns
                if 'result_choice' not in st.session_state:
                    st.session_state['result_choice'] = "Allocation"

                result_choice = st.radio(
                    "Select result view",
                    ("Allocation", "Efficient Frontier", "Risk Analysis", "Model Details", "Portfolio Comparison"),
                    index=(0 if st.session_state['result_choice'] not in ("Allocation", "Efficient Frontier", "Risk Analysis", "Model Details", "Portfolio Comparison") else ["Allocation", "Efficient Frontier", "Risk Analysis", "Model Details", "Portfolio Comparison"].index(st.session_state['result_choice'])),
                    key='result_choice'
                )

                # Allocation view
                if result_choice == "Allocation":
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

                # Efficient frontier view
                if result_choice == "Efficient Frontier":
                    st.subheader("Efficient Frontier & Capital Allocation Line")
                    m = 5
                    N = len(tickers)
                    n_portfolios = max(3000, comb(m + N - 1, N - 1))
                    rand_results = np.zeros((3, n_portfolios))
                    for i in range(n_portfolios):
                        w = np.random.random(len(tickers)); w /= w.sum()
                        rand_results[0, i] = w @ mu.values
                        rand_results[1, i] = np.sqrt(w @ Sigma_percent_sq.values @ w)
                        rand_results[2, i] = (rand_results[0, i] - rf_rate) / rand_results[1, i]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=rand_results[1,:], y=rand_results[0,:], mode='markers', name='Random Portfolios',
                        marker=dict(color=rand_results[2,:], showscale=True, colorscale='viridis', size=5, colorbar=dict(title="Sharpe Ratio"))))
                    fig.add_trace(go.Scatter(x=[opt_result['volatility']], y=[opt_result['return']], mode='markers', name='Optimal Risky Portfolio',
                        marker=dict(color='red', size=12, symbol='star')))
                    fig.add_trace(go.Scatter(x=[allocation['final_volatility']], y=[allocation['final_return']], mode='markers', name='Your Portfolio',
                        marker=dict(color='green', size=12, symbol='diamond')))

                    x_cal = np.linspace(0, opt_result['volatility'] * 1.5, 100)
                    y_cal = rf_rate + opt_result['sharpe'] * x_cal
                    fig.add_trace(go.Scatter(x=x_cal, y=y_cal, mode='lines', name='Capital Allocation Line', line=dict(color='darkorange', width=2, dash='dash')))

                    fig.update_layout(title="Efficient Frontier", xaxis_title="Volatility (Annual %)", yaxis_title="Expected Return (Annual %)", height=500, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    st.plotly_chart(fig, use_container_width=True)

                # Risk analysis view
                if result_choice == "Risk Analysis":
                    st.subheader("Downside Risk Analysis")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Value-at-Risk (95%)", f"{risk_metrics['var_95']:.2f}%", help=f"Your portfolio is not expected to lose more than {abs(risk_metrics['var_95']):.2f}% in a single {st.session_state.frequency} period, with 95% confidence.")
                    col2.metric("Conditional VaR (95%)", f"{risk_metrics['cvar_95']:.2f}%", help=f"In the worst 5% of scenarios, the average loss is expected to be {abs(risk_metrics['cvar_95']):.2f}%.")
                    col3.metric("Max Historical Drawdown", f"{risk_metrics['max_drawdown']:.2f}%", help="The largest peak-to-trough decline observed in the historical data.")

                # Model details view
                if result_choice == "Model Details":
                    st.subheader("Model Inputs & Outputs")
                    st.markdown("### Expected Returns Comparison")
                    mu_hist = calculate_expected_returns(returns, Sigma_decimal, w_market, rf_rate, market_risk_premium, method='historical')
                    mu_eq = calculate_expected_returns(returns, Sigma_decimal, w_market, rf_rate, market_risk_premium, method='equilibrium')

                    comparison_df = pd.DataFrame({
                        'Historical': mu_hist,
                        'Equilibrium': mu_eq,
                        'Difference': mu_hist - mu_eq,
                        'Used': [('✓ (Momentum)' if (returns_method == 'equilibrium' and use_momentum_views) else '✓') if returns_method == 'equilibrium' else '✓' if returns_method == 'historical' else '' for _ in tickers]
                    }, index=tickers)

                    st.dataframe(
                        comparison_df.style.format({
                            'Historical': '{:.2f}%',
                            'Equilibrium': '{:.2f}%',
                            'Difference': '{:.2f}%'
                        }).background_gradient(subset=['Historical', 'Equilibrium'], cmap='RdYlGn', vmin=0, vmax=20),
                        use_container_width=True
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Historical Avg", f"{mu_hist.mean():.2f}%")
                    with col2:
                        st.metric("Equilibrium Avg", f"{mu_eq.mean():.2f}%")
                    with col3:
                        st.metric("Std Dev Difference", f"{(mu_hist.std() - mu_eq.std()):.2f}%")

                    method_used = returns_method.title()
                    if returns_method == 'equilibrium' and use_momentum_views:
                        method_used += " + Momentum Integration"
                    st.info(f"💡 **Current Method**: {method_used}")

                    st.markdown("---")
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

                # Portfolio Comparison view
                if result_choice == "Portfolio Comparison":
                    st.subheader("Portfolio Comparison vs Optimal")
                    st.markdown("Enter your actual portfolio below (one per line: TICKER WEIGHT). We accept weights as decimals (0.25) or percents (25%). We will normalize to 1.0.")
                    actual_text = st.text_area("Your Actual Portfolio", "SPY 0.25\nAAPL 0.05\nBND 0.70", height=180)

                    compare_btn = st.button("Compare Portfolios")
                    if compare_btn:
                        # Keep the UI focused on the Portfolio Comparison view after rerun
                        st.session_state['result_choice'] = 'Portfolio Comparison'
                        actual = parse_actual_portfolio(actual_text)
                        if not actual:
                            st.error("Could not parse actual portfolio. Ensure each line contains 'TICKER WEIGHT' and weights sum to > 0.")
                        else:
                            expanded = sorted(set(tickers) | set(actual.keys()))
                            st.info(f"Expanded universe: {', '.join(expanded)}")

                            missing_tickers = [t for t in expanded if t not in prices.columns]
                            if missing_tickers:
                                with st.spinner("Downloading additional historic data for expanded universe..."):
                                    more_prices = download_data(missing_tickers, start_date, end_date)
                                    if more_prices is None:
                                        st.warning("Could not download data for some tickers; they will be excluded from comparison.")
                                        more_prices = pd.DataFrame()
                                    if not more_prices.empty:
                                        prices = pd.concat([prices, more_prices], axis=1)

                            returns_exp = calculate_returns(prices[expanded].dropna(axis=1, how='all'), st.session_state.frequency)
                            periods_per_year = 52 if st.session_state.frequency == 'weekly' else 12
                            Sigma_percent_sq_exp = returns_exp.cov() * periods_per_year * 10000
                            Sigma_decimal_exp = Sigma_percent_sq_exp.values / 10000

                            mu_exp = {}
                            for t in expanded:
                                if t in mu.index:
                                    mu_exp[t] = mu.loc[t]
                                else:
                                    if t in returns_exp.columns:
                                        mu_exp[t] = float(returns_exp[t].mean() * periods_per_year * 100)
                                    else:
                                        mu_exp[t] = 0.0
                            mu_exp = pd.Series(mu_exp)

                            w_opt = align_weights(expanded, dict(zip(tickers, opt_result['weights'])))
                            w_act = align_weights(expanded, actual)

                            opt_stats = portfolio_stats_from_weights(w_opt, mu_exp.values, Sigma_percent_sq_exp.values, rf_rate)
                            act_stats = portfolio_stats_from_weights(w_act, mu_exp.values, Sigma_percent_sq_exp.values, rf_rate)

                            te_percent, maha = tracking_error_and_mahalanobis(w_act, w_opt, Sigma_decimal_exp)

                            sharpe_gap = act_stats['sharpe'] - opt_stats['sharpe']
                            return_gap = act_stats['return'] - opt_stats['return']
                            vol_gap = act_stats['volatility'] - opt_stats['volatility']
                            score = overall_deviation_score(sharpe_gap, te_percent, return_gap, vol_gap)

                            actual_only = [t for t in actual.keys() if t not in tickers]
                            opt_port_returns = returns_exp @ w_opt
                            subs_df = substitution_analysis(actual_only, returns_exp, mu_exp, opt_port_returns)

                            colA, colB, colC, colD = st.columns(4)
                            colA.metric("Deviation Score", f"{score:.1f}/100")
                            colB.metric("Tracking Error", f"{te_percent:.2f}%")
                            colC.metric("Return Gap", f"{return_gap:.2f}%")
                            colD.metric("Volatility Gap", f"{vol_gap:.2f}%")

                            st.markdown("---")
                            st.subheader("Portfolio Statistics")
                            stats_df = pd.DataFrame({'Optimal': [opt_stats['return'], opt_stats['volatility'], opt_stats['sharpe']],
                                                      'Actual': [act_stats['return'], act_stats['volatility'], act_stats['sharpe']]},
                                                     index=['Return (%)', 'Volatility (%)', 'Sharpe']).T
                            st.dataframe(stats_df.style.format('{:.2f}'), use_container_width=True)

                            st.subheader("Weight Comparison")
                            wc = pd.DataFrame({'Ticker': expanded, 'Optimal Weight': w_opt, 'Actual Weight': w_act})
                            wc['Diff'] = wc['Actual Weight'] - wc['Optimal Weight']
                            wc = wc.set_index('Ticker')
                            st.dataframe((wc*100).style.format('{:.2f}%').background_gradient(subset=['Diff'], cmap='RdYlGn'))

                            if not subs_df.empty:
                                st.subheader("Substitution Analysis (assets only in your actual portfolio)")
                                st.dataframe(subs_df.style.format({'Correlation': '{:.2f}', 'Return (%)': '{:.2f}%', 'Return vs Optimal (%)': '{:.2f}%', 'Quality': '{:.2f}'}), use_container_width=True)

                            st.markdown("---")
                            st.subheader("Interpretation & Limitations")
                            st.markdown("- The Deviation Score is a composite metric (Sharpe penalty ×30, Tracking ×2, Return ×1.5, Volatility ×1). Lower is better.")
                            st.markdown("- Data quality: missing tickers or short histories reduce reliability. The tool uses historical fallbacks when equilibrium returns are unavailable.")
                            st.markdown("- This analysis ignores transaction costs, taxes, and liquidity constraints. Use the score as a heuristic, not financial advice.")
                            st.markdown("- Mahalanobis distance is computed using a pseudo-inverse for numerical stability; interpret as relative, not absolute.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("👈 **Configure your portfolio in the sidebar and click 'Optimize Portfolio' to begin!**")
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1️⃣ Configure Assets")
        st.markdown("""
        - Enter stock/ETF tickers (one per line)
        - Select data history (1-10 years)
        - Choose frequency (weekly/monthly)
        """)
    
    with col2:
        st.markdown("#### 2️⃣ Set Preferences")
        st.markdown("""
        - Enter risk-free rate (e.g., Saving account or T-bill rate)
        - Adjust risk aversion (RA):
          - **RA = 2-4**: Aggressive
          - **RA = 4-6**: Moderate  
          - **RA = 6-10**: Conservative
        """)
    
    with col3:
        st.markdown("#### 3️⃣ Optional Settings")
        st.markdown("""
        - Choose returns method (default: equilibrium)
        - Add custom views for Black-Litterman
        - Click **Optimize Portfolio** button
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 What You'll Get")
        st.markdown("""
        - **Optimal portfolio weights** for each asset
        - **Risk-free allocation** tailored to your RA
        - **Expected return & volatility** metrics
        - **Efficient frontier** visualization
        - **Downside risk analysis** (VaR, CVaR, Max Drawdown)
        """)
    
    with col2:
        st.markdown("### 💡 Understanding the Results")
        st.markdown("""
        The optimizer finds the best mix of risky assets (**tangency portfolio**), then determines how much to allocate between this portfolio and risk-free assets based on your personal risk aversion (RA).
        
        **Formula**: α* = (E[Rₚ] - Rғ) / (RA × σₚ²)
        
        Higher RA → More conservative → More risk-free asset
        """)

st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only and does not constitute financial advice. Data is sourced from Yahoo Finance. Past performance is not indicative of future results.")

# BSD 3-Clause License footer (single block)
st.markdown("Copyright (c) 2025, Omid Arhami. Licensed under the BSD 3-Clause License.")

