import pandas as pd
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly
import plotly.offline as pyo
from prophet import Prophet

# Load the forecast and the original data
forecast_path = r'C:\Users\atall\Documents\atallahchat\Sales_Forecasting\Data\forecast.csv'
df_path = r'C:\Users\atall\Documents\atallahchat\Sales_Forecasting\Data\monthly_sales.csv'

df_forecast = pd.read_csv(forecast_path)
df_monthly = pd.read_csv(df_path)

# Load the trained model (Must match what you trained before)
model = Prophet()
df_monthly = df_monthly.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
model.fit(df_monthly)

# Plot forecast
fig = plot_plotly(model, df_forecast)
pyo.plot(fig)  # Opens the interactive plot in a browser
