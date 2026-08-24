import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Define a function to build a CNN with flexible hyperparameters
def build_custom_cnn(filters=32, dropout_rate=0.2, learning_rate=0.001):
    model = models.Sequential([
        # Convolutional Block 1
        layers.Conv2D(filters, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 2
        layers.Conv2D(filters * 2, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Run Experiments
# Experiment A: Small Filters, High Learning Rate
# Experiment B: Large Filters, Low Learning Rate + Dropout
experiments = [
    {'name': 'Exp A: High LR', 'filters': 32, 'dropout': 0.2, 'lr': 0.01},
    {'name': 'Exp B: Larger/Slower', 'filters': 64, 'dropout': 0.4, 'lr': 0.001}
]

histories = {}

for exp in experiments:
    print(f"\n--- Running {exp['name']} ---")
    model = build_custom_cnn(filters=exp['filters'],
                             dropout_rate=exp['dropout'],
                             learning_rate=exp['lr'])

    h = model.fit(x_train, y_train, epochs=10, batch_size=64,
                  validation_data=(x_test, y_test), verbose=1)
    histories[exp['name']] = h

# 4. Evaluate and Compare Performance
plt.figure(figsize=(12, 5))

for name, h in histories.items():
    plt.plot(h.history['val_accuracy'], label=name)

plt.title('Hyperparameter Comparison: Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
