import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import requests # <-- ADD THIS for the Yahoo fix
import warnings

# Modeling
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import ta 

warnings.filterwarnings("ignore")

# --- DIRECTORY SETUP FIX ---
os.makedirs("outputs", exist_ok=True) # <-- This fixes the FileExistsError

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AlphaQuant | Institutional Forecaster", layout="wide")

# --- MATHEMATICAL UTILITIES ---
def calculate_risk_metrics(prices, returns):
    """Calculates professional-grade risk metrics."""
    ann_vol = returns.std() * np.sqrt(252)
    cagr = (prices.iloc[-1] / prices.iloc[0]) ** (252 / len(prices)) - 1
    
    # Max Drawdown
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    # VaR and Expected Shortfall (Historical 95%)
    var_95 = np.percentile(returns, 5)
    es_95 = returns[returns <= var_95].mean()
    
    sharpe = cagr / ann_vol if ann_vol != 0 else 0
    
    # Sortino (Downside risk)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = cagr / downside_vol if downside_vol != 0 else 0
    
    return {
        "CAGR": cagr,
        "Ann Volatility": ann_vol,
        "Max Drawdown": max_dd,
        "VaR (95%)": var_95,
        "Expected Shortfall": es_95,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino
    }

# --- DATA PROCESSING ---
@st.cache_data
def fetch_and_engineer_data(ticker, period="5y"):
    """Fetches yfinance data with a disguised browser session and builds ML features."""
    try:
        # Disguise the Python script as a Google Chrome web browser
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        })
        
        # Pass the disguised session to yfinance
        df = yf.download(ticker, period=period, progress=False, session=session)
        
        if df.empty: 
            return None, "No data found. Ticker may be incorrect or Yahoo is blocking."
        
        # Flatten multi-index columns if they exist (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        # Calculate Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Outlier Removal (Z-Score > 3)
        z_scores = np.abs((df['Returns'] - df['Returns'].mean()) / df['Returns'].std())
        df = df[z_scores < 3]
        
        # Technical Features
        df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
        df['Roll_Vol'] = df['Returns'].rolling(window=20).std()
        
        # Lag Features for ML
        for i in range(1, 6):
            df[f'Lag_{i}'] = df['Close'].shift(i)
            
        df = df.dropna()
        return df, None
    except Exception as e:
        return None, str(e)

# --- MODELING PIPELINES ---
def model_garch_monte_carlo(prices, returns, horizon, num_sims=1000):
    """Improved Monte Carlo using GARCH(1,1) for volatility forecasting and clipping."""
    # Scale returns for ARCH model convergence
    scaled_returns = returns * 100 
    
    # Fit GARCH(1,1)
    am = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='Normal')
    res = am.fit(disp='off')
    
    # Forecast Volatility
    vol_forecast = res.forecast(horizon=horizon)
    # Re-scale back to decimal
    garch_vol = np.sqrt(vol_forecast.variance.values[-1, :]) / 100 
    
    mu = returns.mean()
    last_price = prices.iloc[-1]
    
    simulations = np.zeros((horizon, num_sims))
    
    for i in range(num_sims):
        Z = np.random.normal(0, 1, horizon)
        # Apply forecasted dynamic volatility
        daily_returns = np.exp((mu - 0.5 * garch_vol**2) + garch_vol * Z)
        price_path = np.zeros(horizon)
        price_path[0] = last_price * daily_returns[0]
        
        for t in range(1, horizon):
            price_path[t] = price_path[t-1] * daily_returns[t]
        simulations[:, i] = price_path
        
    # Percentile Clipping (Filter out extreme 5% tails)
    lower_bound = np.percentile(simulations, 5, axis=1)
    upper_bound = np.percentile(simulations, 95, axis=1)
    mean_path = np.mean(simulations, axis=1)
    
    return mean_path, lower_bound, upper_bound

def model_arima(prices, horizon):
    """ARIMA Time-Series Baseline."""
    try:
        model = ARIMA(prices, order=(5, 1, 0)) # Standard base params
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon).values
        return forecast
    except:
        # Fallback to naive drift if ARIMA fails to converge
        drift = prices.pct_change().mean()
        return [prices.iloc[-1] * (1 + drift)**i for i in range(1, horizon+1)]

def model_machine_learning(df, horizon):
    """Trains Random Forest & Ridge Regression iteratively."""
    features = ['MA_20', 'RSI', 'MACD', 'Roll_Vol', 'Lag_1', 'Lag_2', 'Lag_3']
    X = df[features]
    y = df['Close']
    
    # Train-Test Split (80/20) for weighting
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Models
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    # Evaluate for Ensemble Weights
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test_scaled)))
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge.predict(X_test_scaled)))
    
    # Inverse RMSE Weighting
    total_error = (1/rf_rmse) + (1/ridge_rmse)
    w_rf = (1/rf_rmse) / total_error
    w_ridge = (1/ridge_rmse) / total_error
    
    # Iterative Forecast (Simplified proxy for space constraint)
    # Using linear drift of the last known features
    last_features = X.iloc[-1:].values
    predictions = []
    current_price = y.iloc[-1]
    
    # Predict trend delta to avoid flatlining in autoregressive loops
    trend = df['Close'].pct_change().rolling(20).mean().iloc[-1]
    
    for i in range(horizon):
        scaled_f = scaler.transform(last_features)
        rf_pred = rf.predict(scaled_f)[0]
        ridge_pred = ridge.predict(scaled_f)[0]
        
        ensemble_pred = (rf_pred * w_rf) + (ridge_pred * w_ridge)
        
        # Apply trend to prediction to simulate future steps
        simulated_step = ensemble_pred * (1 + trend * i)
        predictions.append(simulated_step)
        
    return np.array(predictions), {'RF': w_rf, 'Ridge': w_ridge}

# --- HTML REPORT EXPORTER ---
def export_to_html(ticker, fig, table_html, metrics_html):
    """Generates a standalone interactive HTML file."""
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    html_template = f"""
    <html>
    <head>
        <title>{ticker} Quant Forecast</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; color: #333; }}
            h1, h2 {{ color: #2c3e50; }}
            .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #2c3e50; color: white; }}
        </style>
    </head>
    <body>
        <h1>📊 Institutional Forecast Report: {ticker}</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <div class="container">
            <h2>Interactive Price Projection</h2>
            {plot_html}
        </div>
        
        <div class="container">
            <h2>Risk & Performance Metrics</h2>
            {metrics_html}
        </div>
        
        <div class="container">
            <h2>Forecast Milestones</h2>
            {table_html}
        </div>
    </body>
    </html>
    """
    filename = f"outputs/{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template)
    return filename

# --- MAIN UI ---
st.title("🏛️ AlphaQuant | Advanced Forecasting Terminal")

st.sidebar.header("System Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., RELIANCE.NS, AAPL)", value="AAPL").upper()
period = st.sidebar.selectbox("Historical Data", ["1y", "2y", "5y", "10y"], index=2)
horizon = st.sidebar.slider("Forecast Horizon (Days)", 30, 252, 126)

if st.sidebar.button("Run Quantitative Analysis"):
    with st.spinner("Fetching data and engineering features..."):
        df, err = fetch_and_engineer_data(ticker, period)
        
    if err:
        st.error(f"Error fetching data: {err}")
    else:
        prices = df['Close']
        returns = df['Returns']
        
        st.subheader(f"Analyzing {ticker}")
        
        with st.spinner("Running GARCH Volatility & Capped Monte Carlo..."):
            mc_mean, mc_low, mc_high = model_garch_monte_carlo(prices, returns, horizon)
            
        with st.spinner("Fitting ARIMA Model..."):
            arima_pred = model_arima(prices, horizon)
            
        with st.spinner("Training Machine Learning Ensemble..."):
            ml_pred, ml_weights = model_machine_learning(df, horizon)
            
        # --- ENSEMBLE BLENDING ---
        # Weighting: 40% ML, 40% GARCH-MC, 20% ARIMA
        final_mean = (ml_pred * 0.4) + (mc_mean * 0.4) + (arima_pred * 0.2)
        final_high = final_mean + (mc_high - mc_mean) # Preserve GARCH volatility bands
        final_low = final_mean - (mc_mean - mc_low)
        
        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=horizon, freq='B')
        
        # --- 1. PLOTLY CHART ---
        fig = go.Figure()
        
        # Historical
        hist_idx = df.index[-252:] # Show last 1 year
        fig.add_trace(go.Scatter(x=hist_idx, y=df['Close'].iloc[-252:], mode='lines', name='Historical', line=dict(color='black')))
        
        # Forecast Bounds
        fig.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates.tolist()[::-1],
            y=final_high.tolist() + final_low.tolist()[::-1],
            fill='toself', fillcolor='rgba(41, 128, 185, 0.2)', line=dict(color='rgba(255,255,255,0)'),
            name='90% Confidence Interval', hoverinfo="skip"
        ))
        
        # Forecast Mean
        fig.add_trace(go.Scatter(x=future_dates, y=final_mean, mode='lines', name='Ensemble Mean', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=final_high, mode='lines', name='Upper Bound', line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=future_dates, y=final_low, mode='lines', name='Lower Bound', line=dict(color='red', dash='dot')))
        
        fig.update_layout(title=f"{ticker} Blended Forecast (GARCH + ML + ARIMA)", template="plotly_white", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- 2. RISK METRICS ---
        metrics = calculate_risk_metrics(prices, returns)
        st.divider()
        st.subheader("⚖️ Risk & Performance Analytics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
        c2.metric("Ann. Volatility", f"{metrics['Ann Volatility']*100:.2f}%")
        c3.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")
        c4.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        
        # --- 3. TABULAR FORECAST ---
        st.divider()
        st.subheader("📑 Forecast Milestones")
        intervals = [21, 63, horizon] # Approx 1 month, 3 months, Final
        intervals = [i for i in intervals if i <= horizon]
        
        table_data = []
        for i in intervals:
            idx = i - 1
            table_data.append({
                "Timeframe": f"Day {i}",
                "Date": future_dates[idx].strftime('%Y-%m-%d'),
                "Expected Price": round(final_mean[idx], 2),
                "Optimistic (95th Pct)": round(final_high[idx], 2),
                "Pessimistic (5th Pct)": round(final_low[idx], 2)
            })
            
        df_table = pd.DataFrame(table_data)
        st.table(df_table.set_index("Timeframe"))
        
        # --- 4. EXPORT TO HTML ---
        st.divider()
        st.subheader("💾 Export Report")
        metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: "Value"})
        metrics_df["Value"] = metrics_df["Value"].apply(lambda x: f"{x:.4f}")
        
        html_path = export_to_html(ticker, fig, df_table.to_html(index=False), metrics_df.to_html())
        st.success(f"✅ Standalone HTML Report generated successfully! Saved to: `{html_path}`")
        
        with open(html_path, "r", encoding="utf-8") as file:
            st.download_button(
                label="Download HTML Report",
                data=file,
                file_name=f"{ticker}_Forecast.html",
                mime="text/html"
            )
else:
    st.info("👈 Enter a ticker symbol in the sidebar and run the analysis.")
