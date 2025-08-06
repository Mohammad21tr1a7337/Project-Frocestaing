# forecast.py

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

# Set Streamlit config
st.set_page_config(page_title="Energy Consumption Forecasting", layout="wide")

# Background image style
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1581090700227-1e8e5f9a89c1");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"], [data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.8);
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Energy Consumption Forecasting using Holt-Winters")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("PJMW_hourly.csv")
    df.columns = df.columns.str.lower()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.rename(columns={"pjmw_mw": "y", "datetime": "ds"})
    df = df[['ds', 'y']].dropna()
    df = df.set_index('ds').resample('D').mean().dropna()  # Daily average
    return df

df = load_data()
st.subheader("🔢 Sample of Daily Averaged Data")
st.dataframe(df.tail(7))

# Input
days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

# Model
model = ExponentialSmoothing(df['y'], trend="add", seasonal="add", seasonal_periods=7)
fit = model.fit()

# Forecast
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
forecast_values = fit.forecast(days)

# Combine for plotting
past = df.copy()
future = pd.DataFrame({'ds': forecast_index, 'y': forecast_values})
future.set_index('ds', inplace=True)

st.subheader("📈 Forecasted Energy Consumption")
st.dataframe(future.reset_index())

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=past.index, y=past['y'], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=future.index, y=future['y'], mode='lines', name='Forecast'))
fig.update_layout(
    title="Energy Consumption Forecast",
    xaxis_title="Date",
    yaxis_title="Consumption (MW)",
    height=600
)
st.plotly_chart(fig, use_container_width=True)
