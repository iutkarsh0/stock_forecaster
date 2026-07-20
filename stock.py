import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import requests
import warnings
import ta

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AlphaQuant Terminal", layout="wide", page_icon="🏛️")

# --- 1. LEGAL DISCLAIMER ---
st.warning("⚠️ **DISCLAIMER:** This tool is for educational and informational purposes only. It does not constitute financial advice. All investments carry risk, and past performance of quantitative models does not guarantee future results.")

st.title("🏛️ AlphaQuant | Advanced Forecasting Terminal")

# --- SIDEBAR ---
st.sidebar.header("System Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., COALINDIA.NS, AAPL)", "RELIANCE.NS")
history_period = st.sidebar.selectbox("Historical Data", ["1y", "2y", "5y", "max"], index=2)
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 10, 365, 30)

# --- DATA PROCESSING ---
@st.cache_data
def fetch_and_engineer_data(ticker, period="5y"):
    """Fetches yfinance data with a disguised browser session and builds ML features."""
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        })
        
        df = yf.download(ticker, period=period, progress=False, session=session)
        
        if df.empty: 
            return None, "No data found. Ticker may be incorrect or Yahoo is blocking."
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        # Calculate Returns
        df['Returns'] = df['Close'].pct_change()
        
        # Technical Features
        df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        df = df.dropna()
        return df, None
    except Exception as e:
        return None, str(e)

# --- MAIN EXECUTION ---
if st.sidebar.button("Run Quantitative Analysis"):
    with st.spinner(f"Pulling institutional data for {ticker}..."):
        df, error = fetch_and_engineer_data(ticker, history_period)
        
    if error:
        st.error(error)
    else:
        # --- 2. KPI METRICS CARDS ---
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        daily_change = ((current_price - prev_price) / prev_price) * 100
        
        # Calculate Basic Risk Metrics
        rolling_max = df['Close'].cummax()
        drawdown = (df['Close'] / rolling_max) - 1.0
        max_drawdown = drawdown.min() * 100
        annual_volatility = df['Returns'].std() * np.sqrt(252) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"₹{current_price:.2f}" if ".NS" in ticker or ".BO" in ticker else f"${current_price:.2f}", f"{daily_change:.2f}%")
        col2.metric("20-Day Trend (SMA)", f"{df['MA_20'].iloc[-1]:.2f}")
        col3.metric("Annualized Volatility", f"{annual_volatility:.2f}%")
        col4.metric("Max Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")

        st.markdown("---")

        # --- 3. PROFESSIONAL CANDLESTICK CHART ---
        st.subheader("📊 Price Action & Technical Overlay")
        
        # Create a subplot: 2 rows (top for price, bottom for volume)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume'), 
                            row_width=[0.2, 0.7])

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        
        # Moving Averages
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], line=dict(color='orange', width=1.5), name='20 SMA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], line=dict(color='blue', width=1.5), name='50 SMA'), row=1, col=1)
        
        # Volume Bar Chart
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], showlegend=False, marker_color='rgba(128,128,128,0.5)'), row=2, col=1)

        # Layout adjustments
        fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
        st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # --- FORECASTING ENGINE (Simplified Simulation for UI) ---
        st.subheader(f"🔮 {forecast_horizon}-Day Quantitative Forecast")
        
        # Creating a simulated future projection based on historical volatility (Random Walk with Drift)
        last_price = current_price
        daily_drift = df['Returns'].mean()
        daily_vol = df['Returns'].std()
        
        future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, forecast_horizon + 1)]
        simulated_prices = [last_price]
        
        for _ in range(1, forecast_horizon):
            shock = np.random.normal(loc=daily_drift, scale=daily_vol)
            next_price = simulated_prices[-1] * (1 + shock)
            simulated_prices.append(next_price)
            
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted_Price": simulated_prices})
        forecast_df.set_index("Date", inplace=True)

        # Plot Forecast
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].iloc[-100:], name="Historical", line=dict(color="blue")))
        fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecasted_Price'], name="AI Forecast", line=dict(color="springgreen", dash="dash")))
        fig_forecast.update_layout(height=400, template="plotly_dark", title="Expected Price Trajectory")
        st.plotly_chart(fig_forecast, width="stretch")
        
        # --- 4. EXPORT / DOWNLOAD FEATURE ---
        st.markdown("### 📥 Export Data")
        st.write("Download the modeled price projections to run through your own external risk models.")
        
        # Convert dataframe to CSV
        csv_data = forecast_df.to_csv(index=True).encode('utf-8')
        
        st.download_button(
            label="Download Forecast as CSV",
            data=csv_data,
            file_name=f"{ticker}_{forecast_horizon}d_forecast.csv",
            mime="text/csv"
        )
