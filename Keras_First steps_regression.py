# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:54:14 2020

@author: Marcel
"""

import numpy as np
import pandas as pd
import mnist
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

data = pd.read_csv("datasets/cars.csv")
data = data.dropna() # In case any nan are in the dataset

X = data.loc[:, "age":"income"]
y = data.loc[:, "sales"]

# Normalizing X-data
X_stats = X.describe()
X_stats = X_stats.transpose()

def norm(x):
  return (x - X_stats['mean']) / X_stats['std']

X_norm = norm(X)

# Build the model
model = keras.Sequential()
model.add(layers.Dense(100, activation = "sigmoid", input_shape=(5,)))
model.add(layers.Dense(100, activation = "sigmoid"))
model.add(layers.Dense(100, activation = "sigmoid"))
model.add(layers.Dense(100, activation = "sigmoid"))
model.add(layers.Dense(100, activation = "sigmoid"))
model.add(layers.Dense(100, activation = "sigmoid"))
model.add(layers.Dense(1, activation = "sigmoid"))
# Compile
model.compile(
    optimizer = "adam",
    loss = "mse",
    metrics = ["mae", "mse"])
# Fit
model.fit(X, y,
          epochs = 20,
          batch_size=80)
# Predict
predictions = model.predict(X)


# Plot 

plt.scatter(y, predictions)
plt.plot([0,30000], [0,30000], color="red")
plt.show()