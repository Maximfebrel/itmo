import concurrent.futures as cf
from skimage import transform
import numpy as np
from sklearn.model_selection import train_test_split


class Loader:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = self.load_data(r"D:\Хакатоны\im2col\CNNumpy\3. mnist.npz")
        self.x_train = self.resize_dataset(x_train)
        self.x_test = self.resize_dataset(x_test)
        self.y_train = y_train
        self.y_test = y_test

    # загрузка данных
    @staticmethod
    def load_data(path):
        with np.load(path) as f:
            x_train_, y_train_ = f['x_train'], f['y_train']
            x_test_, y_test_ = f['x_test'], f['y_test']
            return (x_train_, y_train_), (x_test_, y_test_)

    # изменение размера датасета для im2col
    @staticmethod
    def resize_dataset(dataset):
        args = [dataset[i:i + 1000] for i in range(0, len(dataset), 1000)]

        def f(chunk):
            return transform.resize(chunk, (chunk.shape[0], 1, 28, 28))

        with cf.ThreadPoolExecutor() as executor:
            res = executor.map(f, args)

        res = np.array([*res])
        res = res.reshape(-1, 1, 28, 28)
        return res

    # разделение данных на батчи
    @staticmethod
    def dataloader(X, BATCH_SIZE, y=None):
        if y is not None:
            n = len(X)
            for t in range(0, n, BATCH_SIZE):
                yield X[t:t + BATCH_SIZE, ...], y[t:t + BATCH_SIZE, ...]
        else:
            n = len(X)
            for t in range(0, n, BATCH_SIZE):
                yield X[t:t + BATCH_SIZE, ...]
