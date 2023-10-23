print("loading Yahoo Finance data...", end='', flush=True)
# Importing yfinance
print("Importing yfinance...", end='', flush=True)
import yfinance as yf
print(" done")

# Importing numpy
print("Importing numpy...", end='', flush=True)
import numpy as np
print(" done")

# Importing pandas
print("Importing pandas...", end='', flush=True)
import pandas as pd
print(" done")

# Importing tensorflow
print("Importing tensorflow...", end='', flush=True)
import tensorflow as tf
print(" done")

# Importing MinMaxScaler from sklearn.preprocessing
print("Importing MinMaxScaler...", end='', flush=True)
from sklearn.preprocessing import MinMaxScaler
print(" done")

# Importing Sequential and Dense from tensorflow.keras.models
print("Importing Sequential and Dense from tensorflow.keras.models...", end='', flush=True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
print(" done")

# Importing matplotlib.pyplot
print("Importing matplotlib.pyplot...", end='', flush=True)
import matplotlib.pyplot as plt
print(" done")
import os
print("!!!!!!!!!!!!!!done!!!!!!!!!!!!!!")
import time
time.sleep(1)
os.system('cls')
# Define the symbol for Intel
symbol = 'INTC'

# Download Intel stock data using yfinance up to the current date
intel_data = yf.download(symbol, start='2020-01-01', end=pd.Timestamp.now().strftime('%Y-%m-%d'))

# Extract the 'Close' prices as our target variable
data = intel_data[['Close']].values

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Define the sequence length (number of days to look back)
sequence_length = 75  # Change this to 10 for the past 10 days

# Create sequences of data
X = []
y = []
for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Build the neural network model
model = Sequential()
model.add(LSTM(units=50, activation='tanh', input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Split the data into training and testing sets
split_index = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=2, validation_split=0.1)

# Make multiple predictions and average them
num_predictions = 32
predicted_prices = []

for _ in range(num_predictions):
    last_sequence = X[-1]
    predicted_price = model.predict(np.array([last_sequence]))
    predicted_price = scaler.inverse_transform(predicted_price)
    predicted_prices.append(predicted_price[0][0])

average_predicted_price = np.mean(predicted_prices)

# Get the current price
current_price = intel_data['Close'].iloc[-1]

# Determine the trend (up or down)
trend = "up" if average_predicted_price > current_price else "down"

# Plot the graph
past_days = intel_data.index[-(sequence_length + 1):-1]
next_day = intel_data.index[-1]

past_prices = intel_data['Close'].iloc[-(sequence_length + 1):-1].values
next_day_prediction = [past_prices[-1], average_predicted_price]

plt.figure(figsize=(10, 6))
plt.plot(past_days, past_prices, color='red', label='Past 10 Days (Close)')
plt.plot([past_days[-1], next_day], next_day_prediction, linestyle='-', marker='o', markersize=8, color='blue', label='Next Day Prediction (Average)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{symbol} Stock Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Print results
print(f"Current {symbol} stock price: {current_price}")
print(f"Average Predicted {symbol} stock price for the next day (averaged over {num_predictions} predictions): {average_predicted_price}")
print(f"Trend: {trend}")
