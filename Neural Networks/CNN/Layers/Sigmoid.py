import numpy as np


class Sigmoid:
    def __init__(self):
        self.y = None

    def feedforward(self, x):
        # вычисление значения на фитфорварде
        x = np.array(x, dtype=np.float32)
        # вычисление значения
        y = 1 / (1 + np.exp(-x) + 10**(-4))
        self.y = y  # для бэкпропа
        return y

    def backprop(self, dE_dy):
        # бэкпроп
        dy_dx = self.y * (1 - self.y)
        dE_dx = dE_dy * dy_dx
        return dE_dx
