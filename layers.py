import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load and prepare data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Define a Deep Architecture
def build_deep_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),

        # Layer 1
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),

        # Layer 2
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),

        # Layer 3
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        # Layer 4
        layers.Dense(64, activation='relu'),

        # Output Layer
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Initialize and Train
deep_net = build_deep_model()
deep_net.summary()

print("\nTraining Deep Neural Network...")
history = deep_net.fit(x_train, y_train,
                        epochs=10,
                        validation_split=0.2,
                        batch_size=128,
                        verbose=1)

# 4. Evaluate and Plot
test_loss, test_acc = deep_net.evaluate(x_test, y_test, verbose=0)
print(f'\nTest Accuracy: {test_acc*100:.2f}%')

# Plot Training Progress
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Deep Network Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
