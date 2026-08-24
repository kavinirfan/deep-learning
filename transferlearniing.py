import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt

# 1. Load and Preprocess Data
# MobileNetV2 expects inputs to be scaled between -1 and 1
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# We use a lambda layer later for resizing, as MobileNet prefers larger images than 32x32
# but we normalize the data here.
x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train)
x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test)

# 2. Load Pre-trained Base Model
# include_top=False removes the final 1000-class classification layer
base_model = applications.MobileNetV2(input_shape=(128, 128, 3),
                                     include_top=False,
                                     weights='imagenet')

# Freeze the base model (do not update its weights during initial training)
base_model.trainable = False

# 3. Build the Fine-Tuning Architecture
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.UpSampling2D(size=(4,4)), # Resize 32x32 to 128x128 for MobileNet
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Phase 1: Train the "Head" (Top layers)
print("Training the new classification head...")
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test), batch_size=64)

# 5. Phase 2: Fine-Tuning (Unfreeze the base model)
print("\nUnfreezing base model for fine-tuning...")
base_model.trainable = True

# We use a much lower learning rate to avoid destroying the pre-trained weights
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(x_train, y_train, epochs=3,
                         validation_data=(x_test, y_test), batch_size=64)
