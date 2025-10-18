import streamlit as st  # Imports Streamlit—lets us build the app with simple commands like st.title() for headers.
import yfinance as yf   # For fetching fresh Bitcoin data—same as notebook.
import pandas as pd     # For data tables—handles DataFrames.
import numpy as np      # For math, like arrays for predictions.
from tensorflow.keras.models import load_model      # To load your saved LSTM model.
import joblib       #To load scalers
from datetime import datetime, timedelta    #For date_calculates e.g.next 7 days
import plotly.graph_objects as go       # Plotly for interactive charts—better than Matplotlib for web (zoom, hover).



# Load the saved LSTM model and scalers.
# Loads the model from file—recreates it ready to predict.
model = load_model('models/lstm_bitcoin_model.keras')
scaler_X = joblib.load('models/scaler_X.pkl')    # Loads feature scaler.
scaler_y = joblib.load('models/scaler_y.pkl')    # Loads target scaler.

# Fetch fresh Bitcoin data—up to today.
ticker = 'BTC-USD'  # Yahoo ticker for Bitcoin.

end_date = datetime.now()  # Today's date.
start_date = end_date - timedelta(days=365)  # Last year for recent context (enough for lags + chart).

data = yf.download(ticker, start=start_date, end=end_date)  # Downloads recent data.
data = data.droplevel(1, axis=1) if isinstance(data.columns, pd.MultiIndex) else data  # Flattens columns if multi-index (like before).

if 'Adj Close' in data.columns:  # Drops Adj Close if present.
    data.drop('Adj Close', axis=1, inplace=True)
data.reset_index(inplace=True)  # Makes Date a column.

# Number of past days the model uses for predictions—match what you tuned in the notebook
# (e.g., change to 30 if that's your best_lag for better accuracy).
lags = 7

# Gets the last 'lags' Close prices from the fresh data as a NumPy array (e.g., last 7 days).
# Why? Model needs recent history to start forecasting the next
# 7—think of it as "what happened lately" to predict ahead.
last_data = data['Close'].values[-lags:]

# Function to predict next 7 days: Uses the model to forecast step by step.
# Takes the loaded model, last real data, scalers, and lags as inputs.
def predict_next_7_days(model, last_data, scaler_X, scaler_y, lags):
    # Starts an empty list to hold the 7 predicted prices—we'll fill it in the loop.
    predictions = []
    # Reshapes the last_data (Close prices) into 3D format (1 sample, 'lags' time steps, 1 feature)—LSTM expects this for sequences.
    current_input = last_data.reshape(1, lags, 1)
    for _ in range(7):  # Loops 7 times, once for each future day.
        # Create a 2D input for scaling: Add a dummy 'Volume' column (0) to match the 8 features the scaler expects.
        # Reshapes current_input to 2D (1 row, lags columns), appends a [0] for dummy Volume (axis=1 adds as new column)
        # —this fixes the shape mismatch.
        input_with_volume = np.append(current_input.reshape(1, -1), [[0]], axis=1)
        # Now scales the full 8-feature input (lags + dummy)—transform expects this shape from training.
        scaled_input = scaler_X.transform(input_with_volume)
        # After scaling, slices only the first 'lags' columns (drops the dummy Volume), reshapes back to 3D for LSTM—this matches how we did in the notebook.
        scaled_input = scaled_input[:, :lags].reshape(1, lags, 1)
        # Uses the model to predict the next price (still scaled)—outputs a 2D array like [[0.5]].
        pred_scaled = model.predict(scaled_input)
        # Inverses the scaling to get real USD price; flatten() turns it into a simple number, [0] grabs the first (only) value.
        pred = scaler_y.inverse_transform(pred_scaled).flatten()[0]
        predictions.append(pred)  # Adds the predicted price to the list.
        # Update current_input for next step: Removes the oldest lag, adds the new prediction (rolling window—like shifting history forward).
        current_input = np.append(current_input[0, 1:], pred).reshape(1, lags, 1)  # np.append joins arrays; [0, 1:] slices off first item.
    return predictions  # Returns the list of 7 predictions after the loop.



# Get the predictions using the function.
# Calls the function with your loaded items; stores the 7 prices in next_7_preds (a list).
next_7_preds = predict_next_7_days(model, last_data, scaler_X, scaler_y, lags)

#Build the App UI (Title, Table, Chart)
# App title and intro text—makes it user-friendly.
st.title('Bitcoin Price Predictor')  # Adds a big header at the top of the app page.
# Adds a short paragraph explaining what the app does—helps users (and resume viewers) understand quickly.
st.write('This app uses a tuned LSTM model to predict Bitcoin prices for the next 7 days based on recent data.')

# Create future dates for the table and chart.
# Makes a list of 7 future dates (e.g., tomorrow to +7 days)—timedelta adds days; i+1 starts from 1.
future_dates = [end_date + timedelta(days=i + 1) for i in range(7)]

# Table of predictions: Shows dates and prices clearly.
# Creates a DataFrame (table) with two columns: Dates and predictions—easy to display.
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close (USD)': next_7_preds})
st.subheader('Predicted Prices for Next 7 Days')  # Adds a smaller header above the table.
st.table(pred_df)  # Displays the table in the app—Streamlit makes it neat, with borders; no need for extra styling.

# Interactive chart: Historical (blue solid) + Predicted (red dashed)—user can distinguish easily.
fig = go.Figure()  # Starts a new Plotly figure (like a blank chart canvas).
# Adds a blue solid line for past prices: x=dates from data, y=Close prices, name for legend.
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Close', line=dict(color='blue')))
# Adds a red dashed line for predictions: x=future dates, y=preds, dash to distinguish from historical.
fig.add_trace(go.Scatter(x=future_dates, y=next_7_preds, mode='lines', name='Predicted Close', line=dict(color='red', dash='dash')))
fig.update_layout(title='Bitcoin Price: Historical (Last Year) and Predicted (Next 7 Days)',  # Sets overall chart title.
                  xaxis_title='Date',  # X-axis label.
                  yaxis_title='Price (USD)',  # Y-axis label.
                  hovermode='x')  # Enables hover: Mouse over shows values for all lines at that date.
st.subheader('Interactive Price Chart')  # Header above the chart.
st.plotly_chart(fig)  # Displays the Plotly chart in Streamlit—users can zoom, pan, hover for details.

st.write('Designed by Shahin Montazeri')



