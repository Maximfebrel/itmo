import numpy as np


class Softmax:
    def __init__(self):
        self.y_l = None

    def feedforward(self, x_l):
        x_l = np.array(x_l, dtype=np.float32)
        y_l = np.exp(x_l) / np.exp(x_l).sum()
        self.y_l = y_l  # для бэкпропа
        return y_l

    def backprop(self, dEdy_l):
        dy_ldx_l = np.zeros((self.y_l.shape[1], self.y_l.shape[1]), dtype=np.float32)
        for i in range(dy_ldx_l.shape[1]):
            for j in range(dy_ldx_l.shape[1]):
                if i == j:
                    dy_ldx_l[i][i] = self.y_l[0][i] * (1 - self.y_l[0][i])
                else:
                    dy_ldx_l[i][j] = - self.y_l[0][i] * self.y_l[0][j]
        dEdx_l = np.dot(dEdy_l, dy_ldx_l)
        return dEdx_l
