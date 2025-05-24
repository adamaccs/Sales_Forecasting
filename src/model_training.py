# src/model_training.py
import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly
import plotly.offline as pyo


def train_model(df):
    """Train the Prophet model."""
    # Check and rename columns for Prophet
    if 'Order Date' not in df.columns or 'Sales' not in df.columns:
        raise ValueError("Error: DataFrame must contain 'Order Date' and 'Sales' columns")

    df_prophet = df.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
    
    # Initialize and fit the model
    model = Prophet()
    model.fit(df_prophet)
    
    return model

def make_forecast(model, periods=12):
    """Make future predictions."""
    future = model.make_future_dataframe(periods=periods, freq='M')  # Ensure monthly frequency
    forecast = model.predict(future)
    return forecast

if __name__ == "__main__":
    # Define absolute path for reliability
    data_path = os.path.abspath(r'C:\Users\atall\Documents\atallahchat\Sales_Forecasting\Data\monthly_sales.csv')
    
    try:
        # Load preprocessed data
        df_monthly = pd.read_csv(data_path)
        
        # Train the model
        model = train_model(df_monthly)
        
        # Make predictions
        forecast = make_forecast(model, periods=12)
        
        # Save the forecast
        forecast_path = os.path.abspath(r'C:\Users\atall\Documents\atallahchat\Sales_Forecasting\Data\forecast.csv')
        forecast.to_csv(forecast_path, index=False)

        print("Forecasting complete. Results saved successfully.")

    except Exception as e:
        print(f"Error: {e}")

# Plots and loads forecast aswell as the original data

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
