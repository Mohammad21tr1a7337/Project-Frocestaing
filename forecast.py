# forecast.py

import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objs as go

st.set_page_config(page_title="⚡ Energy Forecasting (Light)", layout="wide")

st.title("Energy Consumption Forecasting — Holt-Winters")

@st.cache_data
def load_data():
    # Load only last 60 days of data
    df = pd.read_csv("PJMW_hourly.csv", parse_dates=["Datetime"], usecols=["Datetime", "PJMW_MW"])
    df = df.rename(columns={"Datetime":"ds","PJMW_MW":"y"})
    df = df[df["ds"] >= df["ds"].max() - pd.Timedelta(days=60)]
    df = df.set_index("ds").resample("D").mean().dropna()
    return df

df = load_data()
st.subheader("Latest 60 Days (Daily Avg)")
st.dataframe(df.tail())

days = st.number_input("Days to forecast:", min_value=1, max_value=30, value=7)

with st.spinner("🔄 Forecasting..."):
    model = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=7)
    fit = model.fit()
    future_idx = pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=days)
    forecast_vals = fit.forecast(days)

forecast_df = pd.DataFrame({"ds": future_idx, "Forecast": forecast_vals})
st.subheader("Forecast Results")
st.dataframe(forecast_df)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['y'], name="Historical"))
fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["Forecast"], name="Forecast"))
fig.update_layout(xaxis_title="Date", yaxis_title="Energy Consumption", height=500)
st.subheader("Forecast Plot")
st.plotly_chart(fig, use_container_width=True)
