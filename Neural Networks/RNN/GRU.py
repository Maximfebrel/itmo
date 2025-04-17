import numpy as np


class GRU:
    def __init__(self, input_size, output_size, hidden_size=32):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Инициализация весов
        self.Wxhz = np.random.randn(hidden_size, input_size) * 0.01  # Веса для входного слоя
        self.Whhz = np.random.randn(hidden_size, hidden_size) * 0.01  # Веса для скрытого слоя
        self.Wxhr = np.random.rand(hidden_size, input_size) * 0.01
        self.Whhr = np.random.rand(hidden_size, hidden_size) * 0.01
        self.Wxh = np.random.rand(hidden_size, input_size) * 0.01
        self.Whh = np.random.rand(hidden_size, hidden_size) * 0.01

        self.Wy = np.random.randn(output_size, hidden_size) * 0.01  # Веса для выходного слоя

        # Инициализация смещений
        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))

        self.by = np.zeros((output_size, 1))

        self.h = None
        self.z = None
        self.r = None
        self.h_tilde = None
        self.h_prev = np.zeros((hidden_size, 1))
        self.timesteps = None
        self.y_pred = None

    def forward(self, X):
        self.timesteps = X.shape[0]
        self.h = np.zeros((self.timesteps, self.hidden_size))
        self.z = np.zeros((self.timesteps, self.hidden_size))
        self.r = np.zeros((self.timesteps, self.hidden_size))
        self.h_tilde = np.zeros((self.timesteps, self.hidden_size))
        self.y_pred = np.zeros((self.timesteps, self.output_size))  # Массив предсказанных значений

        for t in range(self.timesteps):
            x_t = X[t].reshape(-1, 1)

            # Прямой проход
            self.z[t] = self.sigmoid(np.dot(self.Wxhz, x_t) + np.dot(self.Whhz, self.h_prev) + self.bz).T[0]
            self.r[t] = self.sigmoid(np.dot(self.Wxhr, x_t) + np.dot(self.Whhr, self.h_prev) + self.br).T[0]
            self.h_tilde[t] = np.tanh(np.dot(self.Wxh, x_t) + np.dot(self.Whh, self.r[t] * self.h_prev.T[0]) + self.bh).T[0]

            self.h[t] = (1 - self.z[t]) * self.h_prev.T[0] + self.r[t] * self.h_tilde[t]
            self.h_prev = self.h[t]

            self.y_pred[t] = np.dot(self.Wy, self.h[t]) + self.by

        return self.y_pred

    def backprop(self, X, dloss, lr=None):
        dWxhz = np.zeros_like(self.Wxhz)  # Веса для входного слоя
        dWhhz = np.zeros_like(self.Whhz)  # Веса для скрытого слоя
        dWxhr = np.zeros_like(self.Wxhr)
        dWhhr = np.zeros_like(self.Whhr)
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)

        dWy = np.zeros_like(self.Wy)  # Веса для выходного слоя

        # Инициализация смещений
        dbr = np.zeros_like(self.br)
        dbz = np.zeros_like(self.bz)
        dbh = np.zeros_like(self.bh)

        dby = np.zeros_like(self.by)

        for t in reversed(range(self.timesteps)):
            x_t = X[t].reshape(-1, 1)

            dy = dloss[t]  # Ошибка на выходе

            # Градиенты для весов выходного слоя
            dWy += np.dot(dy, self.h[t].reshape(1, -1))
            dby += dy

            # Обратное распространение ошибки, с учетом активации tanh
            d_h = np.dot(self.Wy.T, dy)

            dz = d_h.T[0] * (self.h_tilde[t] - self.h[t])
            dh_tilde = d_h.T[0] * self.z[t]
            dh_prev = d_h.T[0] * (1 - self.z[t]) * (1 - self.r[t])

            # Градиенты для z_t, r_t, h
            dWxhz += np.dot(np.array([dz]).T, x_t.T)
            dWhhz += np.dot(np.array([dz]).T, np.array([self.h[t]]))
            dbz += np.array([dz]).T

            dWxhr += np.dot(np.array([dh_prev * self.dsigmoid(self.r[t])]).T, x_t.T)
            dWhhr += np.dot(np.array([dh_prev * self.dsigmoid(self.r[t])]).T, np.array([self.h[t]]))
            dbr += np.array([dh_prev * self.dsigmoid(self.r[t])]).T

            dWxh += np.dot(np.array([dh_tilde * self.dtanh(self.h_tilde[t])]).T, x_t.T)
            dWhh += np.dot(np.array([dh_tilde * self.dtanh(self.h_tilde[t])]).T, np.array([self.r[t] * self.h[t]]))
            dbh += np.array([dh_tilde * self.dtanh(self.h_tilde[t])]).T

        self.Wxhr -= lr * dWxhr
        self.Whhr -= lr * dWhhr
        self.Wxhz -= lr * dWxhz
        self.Whhz -= lr * dWhhz
        self.Wxh -= lr * dWxh
        self.Whh -= lr * dWhh
        self.Wy -= lr * dWy

        self.br -= lr * dbr
        self.bz -= lr * dbz
        self.bh -= lr * dbh
        self.by -= lr * dby

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)

    @staticmethod
    def mse(y_pred, y):
        return np.mean(np.square(y_pred[0] - y))

    @staticmethod
    def dmse(y_pred, y):
        n = y.shape[0]
        return (2 / n) * (y_pred.T[0] - y)

    @staticmethod
    def dtanh(dy):
        return 4 * np.exp(2*dy) / (np.exp(2*dy) + 1) ** 2

    def _compute_loss(self, X, y):
        y_pred = self.predict(np.array(X)).T
        return self.mse(y_pred, y)

    def train(self, X, y, lr, epochs=10, batch_size=10):
        epoch_losses = []

        for i in range(epochs):
            # перемешиваем данные в случайном порядке
            # делим генеральную выборку на батчи
            for _ in range(0, len(X), batch_size):
                X_batch = np.array(X[_:_ + batch_size])
                y_batch = np.array(y[_:_ + batch_size])

                # прямой метод
                y_pred = self.forward(X_batch)

                dloss = self.dmse(y_pred, y_batch)
                self.backprop(X_batch, dloss, lr)

            l_ = self._compute_loss(X, y)
            epoch_losses = np.append(epoch_losses, l_)
        return epoch_losses

    def predict(self, X):
        return self.forward(np.array(X))
