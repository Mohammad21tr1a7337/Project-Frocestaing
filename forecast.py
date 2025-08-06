import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="⚡ Energy Forecast", layout="wide")

# Background Image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1581090700227-1e8e5f9a89c1");
    background-size: cover;
    background-position: center;
}
[data-testid="stHeader"], [data-testid="stSidebar"] {
    background: rgba(255,255,255,0.8);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.title("⚡ PJM Energy Consumption Forecasting (Holt-Winters)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("PJMW_hourly.csv", parse_dates=["Datetime"])
    df = df[["Datetime", "PJMW_MW"]].rename(columns={"Datetime": "ds", "PJMW_MW": "y"})
    df = df.set_index("ds").resample("D").mean().dropna()
    return df

df = load_data()

st.subheader("📊 Recent Data")
st.dataframe(df.tail(10))

# User input
days = st.slider("Select number of days to forecast", 1, 60, 15)

# Model
model = ExponentialSmoothing(df['y'], trend="add", seasonal="add", seasonal_periods=7)
fit = model.fit()

forecast = fit.forecast(days)
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
forecast_df = pd.DataFrame({"ds": forecast_index, "Forecast": forecast.values})

# Show forecast
st.subheader("📈 Forecast Table")
st.dataframe(forecast_df)

# Plot
st.subheader("📉 Forecast Chart")
fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(x=df.index, y=df['y'], mode='lines', name='Actual'))

# Forecast
fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['Forecast'], mode='lines', name='Forecast'))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Energy Consumption (MW)",
    height=600,
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)
