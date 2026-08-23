import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load and Preprocess Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Scale pixels to 0-1 range

# 2. Build the Neural Network Architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),    # Input: 28x28 image to 784 vector
    layers.Dense(128, activation='relu'),    # Hidden layer with ReLU activation
    layers.Dropout(0.2),                     # Regularization to prevent overfitting
    layers.Dense(10)                         # Output: 10 units (one per digit)
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4. Train the Model
print("Starting training...")
model.fit(x_train, y_train, epochs=5)

# 5. Evaluate and Predict
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nFinal Test Accuracy: {test_acc*100:.2f}%')

# 6. Visualizing a Prediction
# We add a Softmax layer to convert the model's "logits" into probabilities
probability_model = tf.keras.Sequential([model, layers.Softmax()])
predictions = probability_model.predict(x_test[:1]) # Predict first image

plt.figure(figsize=(4,4))
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Prediction: {np.argmax(predictions[0])}")
plt.axis('off')
plt.show()
