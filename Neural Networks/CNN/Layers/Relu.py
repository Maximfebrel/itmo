import numpy as np


class Relu:
    def __init__(self):
        self.y = None

    def feedforward(self, x):
        # вычисление значения на фитфорварде
        x = np.array(x, dtype=np.float32)
        # вычисление значения
        y = np.where(x > 0, x, 0)
        self.y = y  # для бэкпропа
        return y

    def backprop(self, dE_dy):
        # бэкпроп
        dy_dx = np.where(self.y <= 0, self.y, 1)
        dE_dx = dE_dy * dy_dx
        return dE_dx
