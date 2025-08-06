# forecast.py

import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objs as go

st.set_page_config(page_title="⚡ Energy Forecasting", layout="wide")

# Background style
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1581090700227-1e8e5f9a89c1');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: rgba(0,0,0,0);
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Energy Consumption Forecasting (Holt-Winters)")

@st.cache_data
def load_data():
    df = pd.read_csv("PJMW_hourly.csv", parse_dates=["Datetime"])
    df = df[["Datetime", "PJMW_MW"]].dropna()
    df.columns = ["ds", "y"]
    # Use last 90 days to speed up
    df = df[df["ds"] >= df["ds"].max() - pd.Timedelta(days=90)]
    df = df.set_index("ds").resample("D").mean().dropna()
    return df

df = load_data()
st.subheader("📊 Recent Daily Energy Consumption")
st.dataframe(df.tail(10))

days = st.number_input("📅 Enter number of days to forecast:", min_value=1, max_value=90, value=15)

with st.spinner("🔄 Forecasting using Holt-Winters..."):
    model = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=7)
    fit = model.fit()
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
    forecast_values = fit.forecast(days)

forecast_df = pd.DataFrame({"ds": future_dates, "yhat": forecast_values})

st.subheader("📈 Forecasted Energy Consumption")
st.dataframe(forecast_df)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['y'], name="Actual"))
fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], name="Forecast"))
fig.update_layout(
    title="Energy Forecast vs. Actual",
    xaxis_title="Date",
    yaxis_title="Energy Consumption",
    height=600
)
st.subheader("📉 Forecast Visualization")
st.plotly_chart(fig, use_container_width=True)
