import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf # For downloading real stock data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 1. Download Data (Using Google as an example)
ticker = 'GOOGL'
data = yf.download(ticker, start='2018-01-01', end='2026-01-01')
close_prices = data['Close'].values.reshape(-1, 1)

# 2. Preprocessing
# LSTMs are sensitive to the scale of input data; we scale to (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create a windowed dataset (e.g., use the last 60 days to predict the next day)
prediction_days = 60
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
# Reshape for LSTM: [samples, time_steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 3. Build the LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Train
print(f"Training LSTM on {ticker} data...")
model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)

# 5. Prediction & Visualization
# Get the last 60 days to predict the future
test_inputs = scaled_data[-prediction_days:]
test_inputs = test_inputs.reshape(1, prediction_days, 1)

prediction = model.predict(test_inputs)
prediction = scaler.inverse_transform(prediction) # Convert back to dollars

print(f"\nPredicted price for the next trading day: ${prediction[0][0]:.2f}")

# Plotting historical data
plt.figure(figsize=(10,6))
plt.plot(data['Close'], color='blue', label='Actual Price')
plt.title(f'{ticker} Stock Price History')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.show()
