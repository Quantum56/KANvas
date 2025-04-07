from KANvas import *
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import models, layers
from keras import ops

import KANvas
from KANvas.KANfunctions import ReLU, PLogPhi
from KANvas.KANlayers import KANLayer

test_model = models.Sequential(
    [
        layers.Flatten(),
        KANLayer(32, phi_activation=ReLU, Phi_activation=PLogPhi),
        KANLayer(32, phi_activation=ReLU, Phi_activation=PLogPhi),
        layers.Softmax(10),
    ]
)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

test_model.build(input_shape=x_train.shape)
test_model.compile(optimizer="adamw", loss="categorical_crossentropy")

test_model.fit(ops.convert_to_tensor(x_train), ops.convert_to_tensor(y_train))