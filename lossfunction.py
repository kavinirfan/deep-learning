import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Prepare MNIST Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding (required for MSE classification)
y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
y_test_oh = tf.keras.utils.to_categorical(y_test, 10)

def build_model(loss_function):
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    return model

# 2. Train Model with Mean Squared Error
print("Training with MSE...")
model_mse = build_model('mse')
history_mse = model_mse.fit(x_train, y_train_oh, epochs=10,
                            batch_size=128, validation_data=(x_test, y_test_oh), verbose=0)

# 3. Train Model with Categorical Cross-Entropy
print("Training with Cross-Entropy...")
model_cce = build_model('categorical_crossentropy')
history_cce = model_cce.fit(x_train, y_train_oh, epochs=10,
                            batch_size=128, validation_data=(x_test, y_test_oh), verbose=0)

# 4. Compare Results
plt.figure(figsize=(14, 5))

# Accuracy Comparison
plt.subplot(1, 2, 1)
plt.plot(history_mse.history['val_accuracy'], label='MSE (Test Acc)', color='red', linestyle='--')
plt.plot(history_cce.history['val_accuracy'], label='Cross-Entropy (Test Acc)', color='blue')
plt.title('Accuracy: MSE vs Cross-Entropy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Comparison (Normalized for scale)
plt.subplot(1, 2, 2)
plt.plot(history_mse.history['loss'], label='MSE Loss', color='red', linestyle='--')
plt.plot(history_cce.history['loss'], label='Cross-Entropy Loss', color='blue')
plt.title('Loss Curve: MSE vs Cross-Entropy')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend()

plt.show()
