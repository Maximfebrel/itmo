import numpy as np


class LSTM:
    def __init__(self, input_size, output_size, hidden_size=32):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Инициализация весов
        self.Wxhf = np.random.randn(hidden_size, input_size) * 0.01  # Веса для входного слоя
        self.Whhf = np.random.randn(hidden_size, hidden_size) * 0.01  # Веса для скрытого слоя
        self.Wxhi = np.random.rand(hidden_size, input_size) * 0.01
        self.Whhi = np.random.rand(hidden_size, hidden_size) * 0.01
        self.Wxhc = np.random.rand(hidden_size, input_size) * 0.01
        self.Whhc = np.random.rand(hidden_size, hidden_size) * 0.01
        self.Wxho = np.random.rand(hidden_size, input_size) * 0.01
        self.Whho = np.random.rand(hidden_size, hidden_size) * 0.01

        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Веса для выходного слоя

        # Инициализация смещений
        self.bhf = np.zeros((hidden_size, 1))
        self.bhi = np.zeros((hidden_size, 1))
        self.bhc = np.zeros((hidden_size, 1))
        self.bho = np.zeros((hidden_size, 1))

        self.bhy = np.zeros((output_size, 1))

        self.h = None
        self.h_prev = np.zeros((hidden_size, 1))
        self.ct = None
        self.timesteps = None
        self.y_pred = None
        self.ot = None
        self.ct_ = None
        self.it = None
        self.ft = None

    def forward(self, X):
        self.timesteps = X.shape[0]
        self.h, self.ct = np.zeros((self.timesteps, self.hidden_size)), np.zeros((self.timesteps, self.hidden_size))
        self.y_pred = np.zeros((self.timesteps, self.output_size))  # Массив предсказанных значений
        self.ft = np.zeros((self.timesteps, self.hidden_size))
        self.it = np.zeros((self.timesteps, self.hidden_size))
        self.ct_ = np.zeros((self.timesteps, self.hidden_size))
        self.ot = np.zeros((self.timesteps, self.hidden_size))

        for t in range(self.timesteps):
            x_t = X[t].reshape(-1, 1)

            self.ft[t] = (self.sigmoid(np.dot(self.Wxhf, x_t) + np.dot(self.Whhf, self.h_prev) + self.bhf)).T[0]
            self.it[t] = (self.sigmoid(np.dot(self.Wxhi, x_t) + np.dot(self.Whhi, self.h_prev) + self.bhi)).T[0]
            self.ct_[t] = (np.tanh(np.dot(self.Wxhc, x_t) + np.dot(self.Whhc, self.h_prev) + self.bhc)).T[0]
            self.ct[t] = (self.ft[t] * self.ct[t] + self.it[t] * self.ct_[t]).T

            self.ot[t] = (self.sigmoid(np.dot(self.Wxho, x_t) + np.dot(self.Whho, self.h_prev) + self.bho)).T[0]

            self.h[t] = (self.ot[t] * np.tanh(self.ct[t])).T[0]
            self.h_prev = self.h[t]

            self.y_pred[t] = np.dot(self.Why, self.h[t]) + self.bhy

        return self.y_pred

    def backprop(self, X, dloss, lr=None):
        dWxhf = np.zeros_like(self.Wxhf)  # Веса для входного слоя
        dWhhf = np.zeros_like(self.Whhf)  # Веса для скрытого слоя
        dWxhi = np.zeros_like(self.Wxhi)
        dWhhi = np.zeros_like(self.Whhi)
        dWxhc = np.zeros_like(self.Wxhc)
        dWhhc = np.zeros_like(self.Whhc)
        dWxho = np.zeros_like(self.Wxho)
        dWhho = np.zeros_like(self.Whho)

        dWhy = np.zeros_like(self.Why)  # Веса для выходного слоя

        # Инициализация смещений
        dbhf = np.zeros_like(self.bhf)
        dbhi = np.zeros_like(self.bhi)
        dbhc = np.zeros_like(self.bhc)
        dbho = np.zeros_like(self.bho)

        dbhy = np.zeros_like(self.bhy)

        d_h = np.zeros((self.hidden_size, 1))
        d_c_prev = np.zeros((self.hidden_size, 1))

        for t in reversed(range(self.timesteps)):
            x_t = X[t].reshape(-1, 1)

            h_prev = self.h[t]
            h_t = self.h[t]
            c_prev = self.ct[t]
            f_t = self.ft[t]
            i_t = self.it[t]
            o_t = self.ot[t]
            ct_ = self.ct_[t]

            # Обратное распространение ошибки
            if t == self.timesteps - 1:
                dyt = dloss[t]
            else:
                dyt = np.dot(self.Why, np.array(d_h).T)

            # Градиенты для весов выходного слоя
            dWhy += np.dot(dyt, h_t.reshape(1, -1))
            dbhy += dyt

            d_h += np.dot(self.Why.T, dyt)

            do = d_h.T[0] * np.tanh(c_prev) * self.dsigmoid(o_t)
            d_c = d_c_prev.T[0] + d_h.T[0] * o_t * self.dtanh(c_prev)
            dc_ = d_c * i_t * self.dtanh(ct_)
            di = d_c * ct_ * self.dsigmoid(i_t)
            df = d_c * c_prev * self.dsigmoid(f_t)

            # Обновление весов
            dWxhf += np.dot(np.array([df]).T, x_t.T)
            dWhhf += np.dot(np.array([df]).T, np.array([h_prev]))
            dbhf += np.array([df]).T

            dWxhi += np.dot(np.array([di]).T, x_t.T)
            dWhhi += np.dot(np.array([di]).T, np.array([h_prev]))
            dbhi += np.array([di]).T

            dWxhc += np.dot(np.array([dc_]).T, x_t.T)
            dWhhc += np.dot(np.array([dc_]).T, np.array([h_prev]))
            dbhc += np.array([dc_]).T

            dWxho += np.dot(np.array([do]).T, x_t.T)
            dWhho += np.dot(np.array([do]).T, np.array([h_prev]))
            dbho += np.array([do]).T

            d_h = ([np.dot(self.Whhf.T, df) + np.dot(self.Whhi.T, di) + np.dot(self.Whhc.T, dc_) + np.dot(self.Whho.T, do)])
            d_c_prev = np.array([d_c * f_t]).T

        self.Wxhf -= lr * dWxhf
        self.Whhf -= lr * dWhhf
        self.Wxhi -= lr * dWxhi
        self.Whhi -= lr * dWhhi
        self.Wxhc -= lr * dWxhc
        self.Whhc -= lr * dWhhc
        self.Wxho -= lr * dWxho
        self.Whho -= lr * dWhho
        self.Why -= lr * dWhy

        self.bhf -= lr * dbhf
        self.bhi -= lr * dbhi
        self.bhc -= lr * dbhc
        self.bho -= lr * dbho
        self.bhy -= lr * dbhy

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
            self.ct = np.zeros((batch_size, self.hidden_size))
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
