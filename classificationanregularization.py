import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

# 1. Load and Preprocess MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Build Model with Regularization
def build_regularized_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),

        # Dense layer with L2 Regularization
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),

        # Dropout layer (drops 30% of connections)
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),

        layers.Dropout(0.3),

        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Define Early Stopping Callback
# This stops training if 'val_loss' doesn't improve for 3 straight epochs
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 4. Train the Model
model = build_regularized_model()
print("Training model with L2, Dropout, and Early Stopping...")

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 5. Visualize Training vs Validation
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss with Regularization')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy with Regularization')
plt.legend()

plt.show()
