import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# 1. Load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 2. Define the Autoencoder Architecture
class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim

    # Encoder: Compress the image down to the 'latent_dim'
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(latent_dim, activation='relu'),
    ])

    # Decoder: Reconstruct the image from the compressed vector
    self.decoder = tf.keras.Sequential([
      layers.Dense(128, activation='relu'),
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

# 3. Initialize and Compile
latent_dimension = 32 # 784 pixels compressed into 32 values
autoencoder = Autoencoder(latent_dimension)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# 4. Train the Model
# Notice that the input (x_train) is also the target!
autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

# 5. Visualize the Results
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # Original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
  plt.title("Original")
  plt.axis('off')

  # Reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i], cmap='gray')
  plt.title("Reconstructed")
  plt.axis('off')
plt.show()
