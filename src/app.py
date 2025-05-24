# src/app.py

"""
Sales Forecasting Dashboard with Streamlit and Prophet

How to use:
1. Install dependencies: "pip install -r requirements.txt"
2. Run: "streamlit run src/app.py"
"""

import streamlit as st
import pandas as pd
import os
from prophet import Prophet
import matplotlib.pyplot as plt

# Get the absolute path to the CSV file 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the  current script directory
DATA_PATH = os.path.join(BASE_DIR, "../Data/Superstore.csv")  # Adjusts the relative path

def load_data(file_path):
    """Load the dataset and check if the file exists."""
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found. Please check the path.")
        return None  # Returns None if file is missing
    
    df = pd.read_csv(file_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df_monthly = df.resample('M', on='Order Date').sum().reset_index()
    return df_monthly

def train_model(df):
    """Train the Prophet model."""
    df_prophet = df.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    return model

def main():
    st.title('📈 Sales Forecasting App')

    # Load data
    df_monthly = load_data(DATA_PATH)
    
    if df_monthly is None:
        return  # Stops execution if file is missing

    # Train the model
    model = train_model(df_monthly)

    # Input for number of months to forecast
    months = st.slider('Select number of months to forecast:', 1, 24, 12)

    # Makes predictions
    future = model.make_future_dataframe(periods=months, freq='M')
    forecast = model.predict(future)

    # Plots the forecast
    st.write('## 📊 Sales Forecast')
    fig, ax = plt.subplots()
    model.plot(forecast, ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
