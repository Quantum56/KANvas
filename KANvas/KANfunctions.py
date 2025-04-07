from typing import Callable

if __name__ == "__main__":
    import os
    os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import Layer
from keras import ops
from keras import activations
from keras import initializers
from keras import constraints

class KANActivation:
    pass

@keras.utils.register_keras_serializable(package="KANvas", name="phi")
class phi(Layer, KANActivation):
    """Class for input activation `phi(x)`"""

    def __init__(self, f: Callable, **kwargs):
        super().__init__(**kwargs)
        self.f = f


    def __call__(self, x):
        return self.f(x)

    def call(self, x):
        return self(x)

    def _register(self) -> dict:
        return {"activation": self.f}


@keras.utils.register_keras_serializable(package="KANvas", name="Phi")
class Phi(Layer, KANActivation):
    """Class for unit activation `Phi(x)`"""

    def __init__(self, f: Callable = None, **kwargs):
        super().__init__(**kwargs)
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def call(self, input):
        return self(input)

    def _register(self) -> dict:
        return {"activation": self.f}

@keras.utils.register_keras_serializable(package="KANvas", name="Tanh")
class Tanh(Phi):
    def __init__(self):
        super().__init__(activations.tanh)


@keras.utils.register_keras_serializable(package="KANvas", name="ReLU")
class ReLU(Phi):
    def __init__(self):
        super().__init__(activations.relu)


@keras.utils.register_keras_serializable(package="KANvas", name="LeakyReLU")
class LeakyReLU(Phi):
    def __init__(self):
        super().__init__(activations.leaky_relu)


@keras.utils.register_keras_serializable(package="KANvas", name="GELU")
class GELU(Phi):
    def __init__(self):
        super().__init__(activations.gelu)


@keras.utils.register_keras_serializable(package="KANvas", name="PLog_P")
class PLogPhi(Phi):
    def __init__(self):
        self.a, self.c, self.d = None, None, None
        super().__init__(self.f)

    def build(self, input_shape):
        self.a = self.add_weight(
            shape=input_shape,
            initializer=initializers.Zeros(),
            trainable=True,
        )

        self.c = self.add_weight(
            shape=input_shape,
            initializer=initializers.Zeros(),
            trainable=True,
        )

        self.d = self.add_weight(
            shape=input_shape,
            initializer=initializers.Constant(value=1.0),
            trainable=True,
            constraint=constraints.MaxNorm(max_value=5.0),
        )
        return super().build(input_shape)

    def f(self, x):
        s_d = ops.tanh(self.d)
        
        num = ops.exp(s_d * ops.softplus(self.a) * x) - 1
        denom = 1 - ops.exp(-s_d * ops.softplus(self.c) * x)

        return ops.log(num / denom)


@keras.utils.register_keras_serializable(package="KANvas", name="PLog_p")
class PLog(phi):
    def __init__(self):
        self.a, self.c, self.d = None, None, None

        super().__init__(self.f)

    def build(self, input_shape):
        self.a = self.add_weight(
            shape=input_shape[-1],
            initializer=initializers.Zeros(),
            trainable=True,
        )

        self.c = self.add_weight(
            shape=input_shape[-1],
            initializer=initializers.Zeros(),
            trainable=True,
        )

        self.d = self.add_weight(
            shape=input_shape[-1],
            initializer=initializers.Constant(value=1.0),
            trainable=True,
            constraint=constraints.MaxNorm(max_value=5.0),
        )

        return super().build(input_shape)

    def f(self, x):
        s_d = ops.tanh(self.d)
        
        num = ops.exp(s_d * ops.softplus(self.a) * x) - 1
        denom = 1 - ops.exp(-s_d * ops.softplus(self.c) * x)

        return ops.log(num / denom)


if __name__ == "__main__":
    test_data = ops.convert_to_tensor([1,2,3])
    test = PLog()
    test.build(input_shape=(test_data.shape,))
    print(test(test_data))