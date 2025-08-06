import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objs as go

st.set_page_config(page_title="Energy Consumption Forecasting", layout="wide")

st.markdown(
    '''
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
    ''',
    unsafe_allow_html=True
)

st.title("⚡ Energy Consumption Forecasting (Holt-Winters Model)")

@st.cache_data
def load_data():
    df = pd.read_csv("PJMW_hourly.csv")
    df.columns = df.columns.str.lower()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.rename(columns={"pjmw_mw": "y", "datetime": "ds"})
    df = df[['ds', 'y']].dropna()

    # Reduce to last 6 months for speed
    df = df[df['ds'] >= df['ds'].max() - pd.Timedelta(days=180)]
    df = df.set_index('ds').resample('D').mean().dropna()
    return df

df = load_data()

st.subheader("📊 Recent Energy Consumption (Daily Avg.)")
st.dataframe(df.tail())

days = st.number_input("🔢 Enter number of days to forecast:", min_value=1, max_value=90, value=14)

with st.spinner("Training Holt-Winters model..."):
    model = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=7)
    fit = model.fit()
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
    forecast_values = fit.forecast(days)

st.subheader("📈 Forecasted Energy Consumption")
forecast_df = pd.DataFrame({
    "Date": forecast_index,
    "Forecast": forecast_values
})
st.dataframe(forecast_df)

st.subheader("📉 Energy Consumption Forecast Plot")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['y'], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='Forecast'))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Energy Consumption (MW)',
    height=600,
    legend=dict(x=0, y=1)
)
st.plotly_chart(fig, use_container_width=True)
