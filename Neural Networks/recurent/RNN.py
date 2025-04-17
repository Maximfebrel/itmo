import numpy as np
from numpy.random import randn


class RNN:
    def __init__(self, input_size, output_size, hidden_size=32):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Вес
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Веса для входного слоя
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Веса для скрытого слоя
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Веса для выходного слоя

        # Смещения
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # self.dWxh = np.zeros_like(self.Wxh)
        # self.dWhh = np.zeros_like(self.Whh)
        # self.dWhy = np.zeros_like(self.Why)
        # self.dbh = np.zeros_like(self.bh)
        # self.dby = np.zeros_like(self.by)
        #
        # self.params = {
        #     'wlayx': self.Wxh, 'blayx': self.bh,
        #     'wlayh': self.Whh,
        #     'wlayy': self.Why, 'blayy': self.by,
        # }
        #
        # self.momentum = {}
        # self.rmsprop = {}
        #
        # for key in self.params:
        #     self.momentum['vd' + key] = np.zeros(self.params[key].shape)
        #     self.rmsprop['sd' + key] = np.zeros(self.params[key].shape)

        self.h = None
        self.y_pred = None
        self.h_prev = np.zeros((hidden_size, 1))
        self.timesteps = None

    def forward(self, X):
        self.timesteps = X.shape[0]
        self.h = np.zeros((self.timesteps, self.hidden_size))  # Скрытые состояния
        self.y_pred = np.zeros((self.timesteps, self.output_size))  # Массив предсказанных значений

        for t in range(self.timesteps):
            x_t = X[t].reshape(-1, 1)  # Входной вектор в момент времени t

            # Прямой прогон с использованием конкатенации
            self.h[t] = np.tanh(np.dot(self.Wxh, x_t) + np.dot(self.Whh, self.h_prev) + self.bh).T[0] # Скрытое состояние
            self.y_pred[t] = np.dot(self.Why, self.h[t]) + self.by  # Выходное значение
            self.h_prev = self.h[t]  # Обновление предыдущего состояния

        return self.y_pred

    def backprop(self, X, dloss, lr=None):
        # Инициализация градиентов
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        for t in reversed(range(self.timesteps)):
            dy = dloss[t]  # Ошибка на выходе

            # Градиенты для весов выходного слоя
            dWhy += np.dot(dy, self.h[t].reshape(1, -1))
            dby += dy

            # Обратное распространение ошибки, с учетом активации tanh
            dh = np.dot(self.Why.T, dy)
            dh *= self.dtanh(self.h[t].T[0])  # Получаем градиенты по скрытому состоянию

            # Градиенты для весов скрытого слоя
            dWxh += np.dot(dh, X[t].reshape(1, -1))
            dbh += dh

            if t > 0:
                dWhh += np.dot(dh, self.h[t - 1].reshape(1, -1))

        self.Wxh -= lr * dWxh
        self.Whh -= lr * dWhh
        self.Why -= lr * dWhy
        self.bh -= lr * dbh
        self.by -= lr * dby

        # grads = {
        #     'wlayx': self.dWxh, 'blayx': self.dbh,
        #     'wlayh': self.dWhh,
        #     'wlayy': self.dWhy, 'blayy': self.dby,
        # }
        # return grads

    # def update_params(self, grads, beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.001):
    #     for key in self.params:
    #         self.momentum['vd' + key] = (beta1 * self.momentum['vd' + key]) + (1 - beta1) * grads[key]
    #         self.rmsprop['sd' + key] = (beta2 * self.rmsprop['sd' + key]) + (1 - beta2) * (grads[key] ** 2)
    #         self.params[key] = self.params[key] - (lr * self.momentum['vd' + key]) / (
    #                     np.sqrt(self.rmsprop['sd' + key]) + epsilon)
    #     return self.params
    #
    # def set_params(self, params):
    #     self.Wxh = params['wlayx']
    #     self.bh = params['blayx']
    #
    #     self.Why = params['wlayy']
    #     self.by = params['blayy']
    #
    #     self.Whh = params['wlayh']

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
                # params = self.update_params(grads=grads, lr=lr)
                # self.set_params(params)

            l_ = self._compute_loss(X, y)
            epoch_losses = np.append(epoch_losses, l_)

        return epoch_losses

    def predict(self, X):
        return self.forward(np.array(X))
