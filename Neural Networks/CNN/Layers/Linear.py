import numpy as np

rng = np.random.default_rng(51)


class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self._init_weights()
        self.bias = self._init_biases()
        self.y = None

        self.dbias = 0
        self.dwight = 0

    # задаем начальные значения весом при помощи алгоритма рандомизации, при этом задаем ограничения на генерацию
    def _init_weights(self):
        return np.random.randn(self.out_features, self.in_features) * np.sqrt(1./self.in_features)

    # задаем начальные значения смещения
    def _init_biases(self):
        return np.random.randn(1, self.out_features) * np.sqrt(1./self.out_features)

    def feedforward(self, y):
        # фитфорвард
        x_l = np.dot(y, self.weight.T) + self.bias
        self.y = y  # для бэкпропа
        return x_l

    def backprop(self, dE_dx):
        dE_dw = np.matmul(dE_dx.T, self.y)
        dE_db = np.sum(dE_dx, axis=0)

        dE_dy = np.dot(dE_dx, self.weight)
        self.dwight = dE_dw
        self.dbias = dE_db

        return dE_dy.T, self.dwight, self.dbias


