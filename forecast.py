# forecast.py

import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from PIL import Image
import os

# Page setup
st.set_page_config(page_title="Energy Forecast", layout="wide")

# Background style
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1581090700227-1e8e5f9a89c1");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.8);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.title("⚡ Energy Consumption Forecasting")

# Load data
@st.cache_data
def load_data(file_path="PJMW_hourly.csv"):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    if "Datetime" not in df.columns or "PJMW_MW" not in df.columns:
        st.error("Dataset must contain 'Datetime' and 'PJMW_MW' columns.")
        return pd.DataFrame()
    df.rename(columns={"Datetime": "ds", "PJMW_MW": "y"}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

df = load_data()
if df.empty:
    st.stop()

# Display raw data
st.subheader("📊 Recent Energy Consumption")
st.dataframe(df.tail())

# Forecasting
days = st.number_input("Days to Forecast:", min_value=1, max_value=365, value=30)

model = Prophet()
model.fit(df)

# Future and Forecast
future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)

# Show forecast data
st.subheader("📈 Forecasted Data")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days))

# Plot forecast
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig.update_layout(
    title='Energy Forecast',
    xaxis_title='Date',
    yaxis_title='Energy (MW)',
    height=600
)
st.plotly_chart(fig, use_container_width=True)
