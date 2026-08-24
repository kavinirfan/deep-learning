import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Generate a Sequence (Synthetic "Trend + Noise")
def generate_data(n_steps):
    time = np.arange(0, n_steps, 0.1)
    # Creating a complex wave: Sine + Cosine + Trend
    series = np.sin(time) + 0.5 * np.cos(time * 0.5) + (0.01 * time)
    return series

data = generate_data(1000)

# 2. Sequence Preprocessing
def window_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

WINDOW_SIZE = 20
X, y = window_data(data, WINDOW_SIZE)

# Reshape for LSTM: [batch, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into Train/Test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Build the LSTM Architecture
# We use multiple LSTM layers to create a "Deep" sequence model
model = models.Sequential([
    # Layer 1: return_sequences=True is required to stack another LSTM layer
    layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    layers.Dropout(0.1),

    # Layer 2: return_sequences=False because the next layer is a Dense layer
    layers.LSTM(32, activation='tanh', return_sequences=False),
    layers.Dropout(0.1),

    # Fully Connected output
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 4. Train the Model
print("Training Stacked LSTM...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32,
                    validation_split=0.1, verbose=1)

# 5. Evaluate and Visualize
predictions = model.predict(X_test)

plt.figure(figsize=(12, 5))
plt.plot(y_test, label='Actual Values', color='black', alpha=0.6)
plt.plot(predictions, label='LSTM Predictions', color='red', linestyle='--')
plt.title('Sequence Prediction using Stacked LSTM Architecture')
plt.legend()
plt.show()
