import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Streamlit page settings
st.set_page_config(page_title="Holt-Winters Forecasting", layout="wide")

st.title("⚡ Energy Consumption Forecasting using Holt-Winters")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("PJMW_hourly.csv")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df = df.resample('D').mean()  # aggregate to daily
    df.dropna(inplace=True)
    return df

df = load_data()

# Display last few rows
st.subheader("📊 Historical Energy Consumption Data")
st.dataframe(df.tail())

# Plot actual data
st.line_chart(df, use_container_width=True)

# Forecast input
st.subheader("🔮 Forecast")
days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=90, value=30)

# Fit Holt-Winters model
model = ExponentialSmoothing(df['PJMW_MW'], seasonal_periods=7, trend='add', seasonal='add')
fitted_model = model.fit()

# Forecast
forecast = fitted_model.forecast(steps=days)

# Prepare forecast DataFrame
forecast_df = forecast.reset_index()
forecast_df.columns = ['Datetime', 'Forecast']

# Show forecasted data
st.dataframe(forecast_df.tail())

# Plot forecast
st.subheader("📈 Forecast Plot")
fig, ax = plt.subplots(figsize=(10, 5))
df['PJMW_MW'].plot(ax=ax, label='Historical', linewidth=2)
forecast.plot(ax=ax, label='Forecast', linestyle='--', color='red')
ax.set_xlabel("Date")
ax.set_ylabel("Energy Consumption")
ax.set_title("Energy Consumption Forecast")
ax.legend()
st.pyplot(fig)
