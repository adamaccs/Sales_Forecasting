# src/data_preprocessing.py
import pandas as pd

def load_data(file_path):
    """Load the dataset with the correct encoding."""
    df = pd.read_csv(file_path, encoding="ISO-8859-1")  
    return df

def preprocess_data(df):
    """Preprocess the data."""
    # Convert 'Order Date' to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')  # Handles invalid dates
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['Order Date'])
    
    # Combined sales by month
    df_monthly = df.resample('M', on='Order Date').sum().reset_index()
    
    return df_monthly

if __name__ == "__main__":
    # Load and preprocess data
    file_path = r'C:\Users\atall\Documents\atallahchat\Sales_Forecasting\Data\Superstore.csv'
    
    try:
        df = load_data(file_path)
        df_monthly = preprocess_data(df)

        # Save preprocessed data
        df_monthly.to_csv(r'C:\Users\atall\Documents\atallahchat\Sales_Forecasting\Data\monthly_sales.csv', index=False)
        print("Preprocessing complete. Data saved successfully.")
    
    except Exception as e:
        print(f"Error: {e}")
