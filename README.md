# Bitcoin Price Predictor

This project predicts Bitcoin prices for the next 7 days using machine learning and deep learning models. It includes a Jupyter notebook for modeling and a Streamlit web app for interactive forecasts.

## Project Overview
- **Goal**: Predict BTC-USD prices using baseline, XGBoost, and LSTM models, evaluated with MSE, RMSE, MAPE.
- **Data Source**: Yahoo Finance (via yfinance—free, easy historical prices).
- **Models**: 
  - Baseline: Simple persistence (repeat last price).
  - XGBoost: Machine learning for feature-based predictions.
  - LSTM: Deep learning for time-series sequences (selected as best).
- **App**: Streamlit web app with table and interactive Plotly chart (historical blue, predicted red dashed).

## Setup and Running
1. Clone this repo: `git clone YOUR_GITHUB_LINK`.
2. Install libraries: `pip install -r requirements.txt`.
3. Train models (optional): Open `notebooks/bitcoin_model_training.ipynb` in Jupyter/PyCharm, run all cells to save model/scalers.
4. Run the app: `streamlit run app/bitcoin_predictor_app.py`—opens in browser with predictions.

## Folder Structure
- `notebooks/`: Jupyter file for data, models, evaluation.
- `app/`: Streamlit script.
- `data/`: Saved Bitcoin CSV (historical).
- `models/`: Saved LSTM and scalers.

## Results
- Best model: LSTM with MAPE ~4.68% on test set (accurate for volatile crypto).
- App fetches fresh data for real-time predictions.

## Learning Notes
This project taught data handling, model tuning, evaluation, and deployment. Improvements: Add more features or retrain periodically.

Built by Shahin Montazeri – LinkedIn/GitHub links.