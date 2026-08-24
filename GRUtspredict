import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# 1. Generate Synthetic Time Series Data (Sine Wave)
t = np.linspace(0, 100, 1000)
data = np.sin(t) + np.random.normal(0, 0.1, 1000) # Sine wave + noise

# 2. Prepare the Dataset for a Sequence Model
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

SEQ_LENGTH = 50
X, y = create_sequences(data, SEQ_LENGTH)

# Reshape X for GRU: [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Build the GRU Model
model = Sequential([
    # First GRU layer
    GRU(64, activation='tanh', return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),

    # Second GRU layer (return_sequences=False because the next layer is Dense)
    GRU(32, activation='tanh'),
    Dropout(0.1),

    # Output layer
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 4. Train the Model
print("Training GRU Model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)

# 5. Make Predictions
predictions = model.predict(X_test)

# 6. Visualize Results
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(y_test)), y_test, label='Actual (Test Data)', alpha=0.7)
plt.plot(np.arange(len(predictions)), predictions, label='GRU Prediction', linestyle='--')
plt.title('Time Series Prediction: GRU Model')
plt.legend()
plt.show()
