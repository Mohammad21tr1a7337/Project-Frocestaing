# streamlit_app.py

import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from PIL import Image

# Set page config
st.set_page_config(page_title="Energy Consumption Forecasting", layout="wide")

# Background styling
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
st.title("⚡ Energy Consumption Forecasting with Prophet")

# Upload or load data
@st.cache_data
def load_data():
    # Load your data here, or read from a file
    df = pd.read_csv("energy_consumption.csv")  # change to actual file
    df.rename(columns={"timestamp": "ds", "consumption": "y"}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

df = load_data()
st.subheader("📊 Historical Energy Consumption Data")
st.dataframe(df.tail(10))

# User input
days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

# Prophet model
model = Prophet()
model.fit(df)

# Make future dataframe
future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)

# Show forecasted data
st.subheader("📈 Forecasted Energy Consumption")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days))

# Plot
st.subheader("📉 Forecast Plot")
fig = go.Figure()

# Add past data
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))

# Add forecast
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

# Layout
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Energy Consumption',
    legend=dict(x=0, y=1),
    height=600
)
st.plotly_chart(fig, use_container_width=True)
