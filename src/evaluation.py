# src/evaluation.py
import pandas as pd
from sklearn.metrics import mean_absolute_error

def evaluate_model(actual, predicted):
    """Evaluate the model using MAE."""
    mae = mean_absolute_error(actual, predicted)
    return mae

if __name__ == "__main__":
    # Load actual and predicted data
    df_monthly = pd.read_csv('../data/monthly_sales.csv')
    forecast = pd.read_csv('../data/forecast.csv')

    # Compare actual vs predicted values
    actual = df_monthly['Sales'].tail(12)  # Last 12 months of actual data
    predicted = forecast['yhat'].tail(12)  # Last 12 months of predicted data

    # Evaluate the model
    mae = evaluate_model(actual, predicted)
    print(f'Mean Absolute Error: {mae}')

# Here is the MAE I got"Mean Absolute Error: 10477.081120997116" To know if this value is good we must first calculate the mean sales as comparing it to the MAE reveals if it was good or not.

# Mean Sales Calculation
df_monthly = pd.read_csv('../data/monthly_sales.csv')

mean_sales = df_monthly['Sales'].mean()
mae = 10477  # Use the MAE from your evaluation

relative_mae = (mae / mean_sales) * 100
print(f'Average Sales: {mean_sales:.2f}')
print(f'Relative MAE: {relative_mae:.2f}%')
