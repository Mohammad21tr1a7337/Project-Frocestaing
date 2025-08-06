import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import time

# Set page config
st.set_page_config(page_title="Energy Consumption Forecasting", layout="wide")

# Background styling
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1581090700227-1e8e5f9a89c1');
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

st.title("⚡ Energy Consumption Forecasting with Prophet")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("energy_consumption.csv")
    df.rename(columns={"timestamp": "ds", "consumption": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").tail(500)  # TEMP: Limit to last 500 rows for speed
    return df

df = load_data()
st.subheader("📊 Historical Energy Consumption Data")
st.write(df.tail(10))
st.write(f"✅ Data loaded with shape: {df.shape}")

# User input
days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=365, value=30)

# Train model
with st.spinner("Training the model..."):
    start = time.time()
    model = Prophet()
    model.fit(df)
    st.success(f"Model trained in {round(time.time() - start, 2)} seconds.")

# Make predictions
future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)

# Show forecast results
st.subheader("📈 Forecasted Energy Consumption")
st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days))

# Plot forecast
st.subheader("📉 Forecast Plot")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Actual"))
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Energy Consumption",
    legend=dict(x=0, y=1),
    height=600
)

st.plotly_chart(fig, use_container_width=True)
