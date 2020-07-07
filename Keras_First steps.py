# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:50:08 2020

@author: Marcel
"""

'''

from: https://victorzhou.com/blog/keras-neural-network-tutorial/

The Problem: MNIST digit classification
We’re going to tackle a classic machine learning problem: MNIST handwritten digit classification. 
It’s simple: given an image, classify it as a digit.

'''

import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images -> only one row for each image
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

# Compile the model
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=10,
  batch_size=32,
)