import numpy as np

from utils import im2col, col2im


class MaxPool:
    def __init__(self, kernel, pad=0, stride=2):
        self.kernel = kernel
        self.stride = stride
        self.for_backprop = None
        self.pad = pad

    def feedforward(self, X):
        self.for_backprop = X

        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.pad - self.kernel) / self.stride) + 1
        n_W = int((n_W_prev + 2 * self.pad - self.kernel) / self.stride) + 1

        X_col = im2col(X, self.kernel, self.kernel, self.stride, self.pad)
        X_col = X_col.reshape(n_C, X_col.shape[0] // n_C, -1)
        # находим максимальные элементы
        M_pool = np.max(X_col, axis=1)
        # возвращаем матрицу столбцов к исходному изображению
        M_pool = np.array(np.hsplit(M_pool, m))
        M_pool = M_pool.reshape(m, n_C, n_H, n_W)

        return M_pool

    def backprop(self, de_dx):
        Xprop  = self.for_backprop
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev

        # восстанавливаем исходное изображение
        de_dx_flatten = de_dx.reshape(n_C, -1) / (self.kernel * self.kernel)
        dX_col = np.repeat(de_dx_flatten, self.kernel * self.kernel, axis=0)
        dX = col2im(dX_col, X.shape, self.kernel, self.kernel, self.stride, self.pad)
        # приводим изображение к необходимому размеру
        dX = dX.reshape(m, -1)
        dX = np.array(np.hsplit(dX, n_C_prev))
        dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)
        return dX
