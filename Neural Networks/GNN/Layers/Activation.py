import numpy as np


class Sigmoid:
    def __init__(self):
        self.x = None

    def __call__(self, *args, **kwargs):
        self.x = args[0]
        return 1 / (1 + np.exp(-self.x))

    def backward(self):
        return self.x * (1 - self.x)


class Relu:
    def __init__(self):
        self.x = None

    def __call__(self, *args, **kwargs):
        self.x = args[0]
        return np.maximum(0, self.x)

    def backward(self, dout):
        dout[self.x <= 0] = 0
        return dout


class Tanh:
    def __init__(self):
        self.input = None

    def __call__(self, *args, **kwargs):
        out = np.tanh(args[0])
        return out

    def backward(self, dout):
        grad = 1 - self.input ** 2
        return grad * dout


class Loss:
    def __init__(self):
        self.y_pred = None
        self.y = None

    def __call__(self, *args, **kwargs):
        self.y = args[0]
        self.y_pred = args[1]
        return np.mean(np.square(self.y_pred - self.y[:, None]))

    @staticmethod
    def backward(y_pred, y):
        n = y.shape[0]
        return (2 / n) * (y_pred - y[:, None])
