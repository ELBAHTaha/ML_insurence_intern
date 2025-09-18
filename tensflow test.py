import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

X = np.random.rand(100,5)
y = np.random.randint(0,2,100)

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=5, verbose=1)
print("Done")
