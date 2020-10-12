# test.py
# Copyright (c) 2020, zhiayang
# Licensed under the Apache License Version 2.0.

import keras
import numpy as np

from keras import callbacks, regularizers
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

np.random.seed(1)

inp = Input(shape=(2,))
x = Dense(10, activation="sigmoid")(inp)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)


x_train = np.repeat([[0, 0], [0, 1], [1, 0], [1, 1]], 200, axis=0)
y_train = np.repeat([0, 1, 1, 0], 200, axis=0)

model.compile(loss="mse", metrics=["acc"], optimizer=Adam(learning_rate=0.01))

print(model.summary())

history = model.fit(x_train, y_train,
	batch_size=2, epochs=25, verbose=1
)

print("0 ^ 0 = {}".format(model.predict(np.array([[0, 0]]))))
print("0 ^ 1 = {}".format(model.predict(np.array([[0, 1]]))))
print("1 ^ 0 = {}".format(model.predict(np.array([[1, 0]]))))
print("1 ^ 1 = {}".format(model.predict(np.array([[1, 1]]))))
