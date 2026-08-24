import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Simulate a Preprocessed & Engineered Dataset
# Let's assume you have 20 engineered features (e.g., mean, variance, etc.)
num_samples = 5000
num_features = 20
num_classes = 3

# Generating dummy preprocessed data
x_data = np.random.random((num_samples, num_features))
# Creating target labels (0, 1, or 2)
y_data = np.random.randint(0, num_classes, num_samples)

# Split into train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

# 2. Build the Simple Neural Network
def build_simple_model(input_dim):
    model = models.Sequential([
        # Input layer: matched to the number of engineered features
        layers.Input(shape=(input_dim,)),

        # First hidden layer
        layers.Dense(64, activation='relu'),

        # Second hidden layer
        layers.Dense(32, activation='relu'),

        # Output layer (3 neurons for 3 classes)
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Initialize and Train
model = build_simple_model(num_features)
model.summary()

print("\nTraining on engineered features...")
history = model.fit(x_train, y_train,
                    epochs=15,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=1)

# 4. Evaluate Performance
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\nFinal Test Accuracy: {test_acc*100:.2f}%')

# 5. Plot Accuracy Curve
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
