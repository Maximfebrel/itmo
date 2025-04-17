import numpy as np

from utils import im2col, col2im


class Conv:
    def __init__(self, kernel, in_channels, out_channels, stride, pad=0):
        # задаем размер ядра
        self.for_backprop = None
        self.kernel = kernel
        # задаем количество входных каналов
        self.in_channels = in_channels
        # задаем количество выходных каналов
        self.out_channels = out_channels
        # задаем шаг
        self.stride = stride
        self.pad = pad

        # задаем начальные веса и смещения
        self.weight = self.do_init_weights()
        self.bias = self.do_init_bias()
        self.dbias = 0
        self.dwight = 0

    # задание начальных весов
    def do_init_weights(self):
        # задаем начальные веса для ядра свертки, при этом для каждого выходного канала задаем свое ядро
        return np.random.randn(self.out_channels, self.in_channels, self.kernel, self.kernel) * np.sqrt(1. / self.kernel)

    # задание начальных смещений
    def do_init_bias(self):
        # задаем смещение, при этом для каждого выходного канала задаем свое смещение
        return np.random.randn(self.out_channels) * np.sqrt(1. / self.out_channels)

    # умножение на ядро
    def feedforward(self, X):
        m, n_c_prev, n_h_prev, n_w_prev = X.shape

        # количество выходных каналов
        n_channels = self.out_channels
        # высота и ширина свертки
        n_height = int((n_h_prev + 2 * self.pad - self.kernel)) + 1
        n_weight = int((n_w_prev + 2 * self.pad - self.kernel)) + 1

        # выполняем преобразование входного изображения в столбцы, так, чтобы можно было умножать эту матрицу на окно
        X_col = im2col(X, self.kernel, self.kernel, 1, self.pad)
        # приводим веса и смещение к виду, чтобы можно было умножать
        w_col = self.weight.reshape((self.out_channels, -1))
        b_col = self.bias.reshape(-1, 1)

        # выполняем матричное умножение
        out = w_col @ X_col + b_col

        # выпоплняем преобразование полученной матрицы столбцов в изображение для дальнейшей работы
        out = np.array(np.hsplit(out, m)).reshape((m, n_channels, n_height, n_weight))
        self.for_backprop = X, X_col, w_col
        return out

    def backprop(self, de_dx):
        X, X_col, w_col = self.for_backprop
        m, _, _, _ = X.shape
        # обновляем смещение
        self.dbias = np.sum(de_dx, axis=(0, 2, 3)) 

        # приводим выход с предыдущего слоя к необходимому виду (для col2im)
        de_dx = de_dx.reshape(de_dx.shape[0] * de_dx.shape[1], de_dx.shape[2] * de_dx.shape[3])
        de_dx = np.array(np.vsplit(de_dx, m))
        de_dx = np.concatenate(de_dx, axis=-1)

        # получаем матрицу столбцов как выход слоя и приводим ее к виду изображения
        dX_col = w_col.T @ de_dx
        dX = col2im(dX_col, X.shape, self.kernel, self.kernel, 1, self.pad)

        # обновляем веса
        dw_col = de_dx @ X_col.T
        self.dwight = dw_col.reshape((dw_col.shape[0], self.in_channels, self.kernel, self.kernel)) 

        return dX, self.dwight, self.dbias
