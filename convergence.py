import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Prepare Data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# For Softmax, we typically use SparseCategoricalCrossentropy
# For Sigmoid to work in a multi-class setting, we often treat it as 10 binary problems
# We'll convert labels to one-hot encoding for a fair comparison
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

def build_model(output_activation):
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation=output_activation)
    ])

    # We use BinaryCrossentropy for Sigmoid and CategoricalCrossentropy for Softmax
    loss_fn = 'binary_crossentropy' if output_activation == 'sigmoid' else 'categorical_crossentropy'

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

# 2. Train both models
epochs = 10
print("Training Model with Sigmoid Output...")
sigmoid_model = build_model('sigmoid')
history_sigmoid = sigmoid_model.fit(x_train, y_train_one_hot, epochs=epochs,
                                    validation_data=(x_test, y_test_one_hot), verbose=0)

print("Training Model with Softmax Output...")
softmax_model = build_model('softmax')
history_softmax = softmax_model.fit(x_train, y_train_one_hot, epochs=epochs,
                                    validation_data=(x_test, y_test_one_hot), verbose=0)

# 3. Plotting the Results
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_sigmoid.history['accuracy'], label='Sigmoid Train Acc', linestyle='--')
plt.plot(history_softmax.history['accuracy'], label='Softmax Train Acc')
plt.title('Convergence: Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history_sigmoid.history['loss'], label='Sigmoid Train Loss', linestyle='--')
plt.plot(history_softmax.history['loss'], label='Softmax Train Loss')
plt.title('Convergence: Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
