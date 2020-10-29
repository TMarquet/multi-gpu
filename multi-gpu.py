
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

import os
tf.enable_eager_execution()
DEVICES = ['/device:GPU:0','/device:GPU:1']
strategy = tf.distribute.MirroredStrategy(devices= DEVICES)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0

test_images = test_images / 255.0

with strategy.scope():
  input_re = keras.layers.Flatten(input_shape=(28, 28))
  hid = keras.layers.Dense(128, activation='relu')
  out =  keras.layers.Dense(10)
  model = keras.Sequential([
  input_re,
  hid,
  out
  ])

  model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# Define the checkpoint directory to store the checkpoints



model.fit(train_images, train_labels, epochs=10)
