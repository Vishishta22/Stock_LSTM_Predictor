import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# App configuration
st.set_page_config(layout="wide", page_title="Apple Stock Analyzer")

# Constants
ticker = "AAPL"
today = datetime.today().date()

# Caching data
@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# Session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

# ------------------- HOME PAGE -------------------
if st.session_state.page == "home":
    st.markdown("<h1 style='text-align: center;'> Apple Stock Analysis & Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("###", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 4, 2])
    with col2:
        st.markdown("#### Choose what you'd like to explore:")
        if st.button("📈 Historical Analysis"):
            st.session_state.page = "historical"
        if st.button("🔮 Future Forecasting"):
            st.session_state.page = "forecasting"

# ------------------- HISTORICAL ANALYSIS -------------------
elif st.session_state.page == "historical":
    st.header("📈 Historical Analysis of Apple Stock")
    if st.button("⬅️ Back to Home"):
        go_home()

    st.markdown("Select a date range from **2005-01-01 to yesterday** for historical analysis.")

    start_date = st.date_input("Start Date", datetime(2005, 1, 1).date(), min_value=datetime(2005, 1, 1).date(), max_value=today - timedelta(days=1))
    end_date = st.date_input("End Date", today - timedelta(days=1), min_value=datetime(2005, 1, 1).date(), max_value=today - timedelta(days=1))

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    df = load_data(ticker, start=start_date, end=end_date + timedelta(days=1))
    if df.empty:
        st.warning("No data available for selected range.")
        st.stop()

    st.subheader("Analysis Tools")

    if st.button("Show Summary Statistics"):
        st.subheader("🧾 Summary Statistics")
        st.dataframe(df.describe())

    if st.button("Show Line Chart"):
        st.subheader("📊 Closing Price Over Time")
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'], label='Close Price')
        ax.set_title("Apple Closing Price")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)
        st.markdown("**Inference:** The line chart shows overall price trends, indicating bullish or bearish phases based on slope and fluctuation.")

    if st.button("Show Moving Averages"):
        st.subheader("📉 Moving Averages")
        short_window = st.slider("Short-term MA (days)", 5, 50, 20)
        long_window = st.slider("Long-term MA (days)", 20, 200, 100)
        df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
        df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'], label='Close', alpha=0.4)
        ax.plot(df.index, df['Short_MA'], label=f'{short_window}-Day MA')
        ax.plot(df.index, df['Long_MA'], label=f'{long_window}-Day MA')
        ax.set_title("Moving Averages")
        ax.legend()
        st.pyplot(fig)
        st.markdown("**Inference:** Crossovers between short and long MAs are typical buy/sell indicators for traders.")

    if st.button("Show RSI and MACD"):
        st.subheader("📐 RSI and MACD Indicators")
        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain.flatten()).rolling(window=14).mean()
        avg_loss = pd.Series(loss.flatten()).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        df['12ema'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['26ema'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['12ema'] - df['26ema']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax1.plot(df.index, rsi, label='RSI')
        ax1.axhline(70, color='red', linestyle='--')
        ax1.axhline(30, color='green', linestyle='--')
        ax1.set_title('RSI Indicator')

        ax2.plot(df.index, df['MACD'], label='MACD')
        ax2.plot(df.index, df['Signal'], label='Signal Line')
        ax2.set_title('MACD Indicator')
        ax1.legend()
        ax2.legend()
        st.pyplot(fig)
        st.markdown("**Inference:** RSI >70 suggests overbought, <30 oversold. MACD crossing signal line may indicate reversals.")

# ------------------- FORECASTING -------------------
elif st.session_state.page == "forecasting":
    st.header("🔮 Future Stock Price Forecasting")
    if st.button("⬅️ Back to Home"):
        go_home()

    df = load_data(ticker, start="2005-01-01", end=today)
    data = df[['Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    window_size = 60
    last_sequence = scaled_data[-window_size:]

    try:
        model = load_model("lstm_model.h5")
        st.success("You are ready to go")
    except Exception as e:
        st.error("Failed to load pre-trained model. Please ensure 'lstm_model.h5' is present.")
        st.stop()

    st.markdown("### Select forecast date range")
    future_start = st.date_input("Forecast Start Date", min_value=today + timedelta(days=1))
    future_end = st.date_input("Forecast End Date", min_value=future_start)

    if future_start <= today or future_end <= today:
        st.error("Please choose future dates only.")
        st.stop()

    forecast_days = (future_end - future_start).days + 1

    
    plot_clicked = st.button("Plot Forecast", key="plot_button_main")
    table_clicked = st.button("Show Forecast Table", key="table_button_main")

    if plot_clicked or table_clicked:
        predicted = []
        temp_sequence = last_sequence.copy()
        for _ in range(forecast_days):
            pred_input = temp_sequence.reshape((1, window_size, 1))
            pred = model.predict(pred_input, verbose=0)
            predicted.append(pred[0, 0])
            temp_sequence = np.append(temp_sequence[1:], pred)

        predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1, 1))
        future_dates = pd.date_range(future_start, periods=forecast_days)

        if plot_clicked or table_clicked:
            st.subheader("📈 Forecasted Prices")
            fig, ax = plt.subplots()
            ax.plot(future_dates, predicted_prices, label="Forecast", color="orange", linestyle='dashed', marker='o')
            ax.set_title("Forecasted Apple Prices")
            ax.set_ylabel("Price (USD)")
            ax.set_xlabel("Date")
            ax.legend()
            # Format x-axis to prevent overlapping
            fig.autofmt_xdate()  # Automatically rotates dates
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Adjusts tick frequency
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set readable date format
            plt.xticks(rotation=45)  # Optional: rotate further for better readability
            st.pyplot(fig)

            if table_clicked:
                st.subheader("📊 Forecasted Prices Table")
                forecast_df = pd.DataFrame({"Date": future_dates.date, "Predicted Price": predicted_prices.flatten()})
                st.dataframe(forecast_df)
                st.download_button("Download Forecast CSV", forecast_df.to_csv(index=False), file_name="forecasted_prices.csv")

            # Inference
            st.markdown("### 📌 Inference")
            trend = "increasing" if predicted_prices[-1] > predicted_prices[0] else "decreasing"
            percent_change = ((predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0]) * 100
            st.markdown(f"🔍 Expected change over selected period: **{float(percent_change):.2f}%**")

            suggestion = (
                "📈 The forecast shows an **upward trend**, which may be a good opportunity to consider **investing** in Apple stock." 
                if trend == "increasing"
                else "📉 The forecast suggests a **downward trend**, so it might be wise to **wait** before investing."
            )
            st.markdown(suggestion)
