import numpy as np


class LayerNorm:
    def __init__(self, gamma, beta, eps):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.cache = None

    def feedforward(self, x):
        N, D = x.shape
        # вычисление среднего
        mu = 1. / N * np.sum(x, axis=0)
        xmu = x - mu
        sq = xmu ** 2
        # вычисление дисперсии
        var = 1. / N * np.sum(sq, axis=0)
        # корень из дисперсии плюс отклонение
        sqrtvar = np.sqrt(var + self.eps)
        # вычисляем обратную величину
        ivar = 1. / sqrtvar
        # нормализуем вход
        xhat = xmu * ivar
        # масштабируем
        gammax = self.gamma * xhat
        # смещаем
        out = gammax + self.beta
        # для бэкпропа
        self.cache = (xhat, self.gamma, xmu, ivar, sqrtvar, var, self.eps)
        return out

    def backward(self, dout):
        xhat, gamma, xmu, ivar, sqrtvar, var, eps = self.cache
        # размерность входа
        N, D = dout.shape
        # вычисляем градиент для beta
        dbeta = np.sum(dout, axis=0)
        dgammax = dout
        # вычисляем градиент для гама
        dgamma = np.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * gamma
        # вычисление градиентов для обратной величины и среднего
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar
        # градиент для корня
        dsqrtvar = -1. / (sqrtvar ** 2) * divar
        # градиент для дисперсии
        dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar
        # градиент для квадрата
        dsq = 1. / N * np.ones((N, D)) * dvar
        # градиент для среднего
        dxmu2 = 2 * xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
        dx2 = 1. / N * np.ones((N, D)) * dmu
        # вычисление финального градиента
        dx = dx1 + dx2

        self.gamma -= dgamma
        self.beta -= dbeta
        return dx
