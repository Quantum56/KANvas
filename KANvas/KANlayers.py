import keras
from keras import layers
from keras import models
from keras import ops
import numpy as np

from .KANfunctions import *


class KANCell(layers.Layer):
    def __init__(
        self,
        #input_shape: tuple = None,
        phi_activation: object = ReLU,
        Phi_activation: object = ReLU,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.phi_activation = phi_activation

        self.phi_activations: list = []

        self.Phi_activation = Phi_activation

    def build(self, input_shape):
        self.phi_activations = [
            self.phi_activation() for _ in range(np.prod(input_shape[1:]))
        ]

        self.Phi_activation = self.Phi_activation()

        for i in range(np.prod(input_shape[1:])):
            self.phi_activations[i].build((1,))
        self.Phi_activation.build((input_shape[-1], ))

        super().build(input_shape)

    def call(self, inputs):
        res = self.apply_activation_list(inputs, self.phi_activations)
        res = ops.sum(res)
        res = self.Phi_activation(res)
        return res

    @staticmethod
    def apply_activation_list(inputs, activations):
        # inputs: Tensor of shape (N,)
        # activations: list of callables (e.g., keras.layers.Layer or functions)
        outputs = ops.stack(
            [activations[i](inputs[i]) for i in range(len(activations))]
        )
        return outputs


class KANLayer(layers.Layer):
    def __init__(self, units: int, phi_activation=ReLU, Phi_activation: KANActivation = PLogPhi, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.phi_activation_cls = phi_activation
        self.Phi_activation_cls = Phi_activation
        self.cells = []

    def build(self, input_shape):
        for _ in range(self.units):
            phi = self.phi_activation_cls
            Phi = self.Phi_activation_cls
            cell = KANCell(phi, Phi)
            cell.build(input_shape)
            self.cells.append(cell)

    def call(self, inputs):
        outputs = [cell(inputs) for cell in self.cells]
        return ops.convert_to_tensor(outputs)

if __name__ == "__main__":
    import os
    os.environ["KERAS_BACKEND"] = "jax"

    test_model = models.Sequential(
        [
            KANLayer(32, phi_activation=ReLU, Phi_activation=PLogPhi),
            KANLayer(32, phi_activation=ReLU, Phi_activation=PLogPhi),
            layers.Dense(1),
        ]
    )

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = ops.reshape(x_train, (-1,))
    x_test = ops.reshape(x_train, (-1,))

    test_model.fit(x_train, y_train)
