# Sales Forecasting with Prophet

Basic Streamlit app that predicts retail sales from Superstore data.

Here is the link for the 'Superstore.csv' data file:"https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting"

## Quick Start
1. Clone this repo
2. Run: `pip install -r requirements.txt`
3. Launch: `streamlit run src/app.py`

## Files
- `app.py` - Main dashboard (runs the forecast)
- `data_preprocessing.py` - Cleans the CSV data
- `model_training.py` - Prophet model code
- `evaluation.py` - Checks model accuracy (MAE: ~10,477)

## Requirements
- Python 3.9+
- Libraries in requirements.txt

## Data
Uses default Superstore.csv (sales from 2014-2017). Change the path in `app.py` if your CSV is elsewhere.

## Known Issues
- First run might be slow (Prophet compiling)
- Doesn't handle missing data well
