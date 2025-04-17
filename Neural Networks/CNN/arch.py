import numpy as np
import pandas as pd

from Layers.Max import MaxPool
from Layers.Conv import Conv
from Layers.Linear import Linear
from Layers.Relu import Relu
from Layers.Sigmoid import Sigmoid
from Layers.Softmax import Softmax
from data_loader import Loader


class CNN:
    def __init__(self):
        self.sigmoid1_shape = None
        self.conv1 = Conv(5, 1, 4, 1)
        self.relu1 = Relu()

        self.max1 = MaxPool(2)

        self.conv2 = Conv(5, 4, 12, 1)
        self.relu2 = Relu()

        self.max2 = MaxPool(2)
        # self.conv3 = Conv(4, 12, 26, 1)

        self.max2_shape = None
        # self.conv3_shape = None
        # self.sigmoid1 = Sigmoid()

        # self.lay1 = Linear(26, 10)
        # self.sigmoid1 = Sigmoid()
        self.lay2 = Linear(192, 64)
        self.sigmoid2 = Sigmoid()

        self.lay3 = Linear(64, 10)
        self.softmax = Softmax()

        self.loss = Loss()

        self.layers = [self.conv1, self.conv2, self.lay2, self.lay3] # self.conv3, self.lay1] # , self.lay3, self.lay2, self.lay3]

    def feedforward(self, X):
        conv1 = self.conv1.feedforward(X)
        relu1 = self.relu1.feedforward(conv1)

        max1 = self.max1.feedforward(relu1)

        conv2 = self.conv2.feedforward(max1)
        relu2 = self.relu2.feedforward(conv2)

        max2 = self.max2.feedforward(relu2)

        # conv3 = self.conv3.feedforward(max2)
        # sigmoid1 = self.sigmoid1.feedforward(conv3)

        # self.sigmoid1_shape = sigmoid1.shape
        # sigmoid1_flatten = sigmoid1.reshape(self.sigmoid1_shape[0], -1)

        self.max2_shape = max2.shape  # для бэкпропа

        max2_flatten = max2.reshape(self.max2_shape[0], -1)
        # lay1 = self.lay1.feedforward(max2_flatten)
        # sigmoid1 = self.sigmoid1.feedforward(lay1)

        lay2 = self.lay2.feedforward(max2_flatten)
        sigmoid2 = self.sigmoid2.feedforward(lay2)

        lay3 = self.lay3.feedforward(sigmoid2)
        y = self.softmax.feedforward(lay3)
        return y

    def backprop(self, y):
        y = self.softmax.backprop(y)
        y, wlay3, blay3 = self.lay3.backprop(y)

        y = self.sigmoid2.backprop(y.T)
        y, wlay2, blay2 = self.lay2.backprop(y)

        # y = self.sigmoid1.backprop(y.T)
        # y, wlay1, blay1 = self.lay1.backprop(y)

        y = y.reshape(self.sigmoid1_shape)
        # y = self.sigmoid1.backprop(y)
        #
        # y, wconv3, bconv3 = self.conv3.backprop(y)

        y = self.max2.backprop(y)

        y = self.relu2.backprop(y)
        y, wconv2, bconv2 = self.conv2.backprop(y)

        y = self.max1.backprop(y)

        y = self.relu1.backprop(y)
        y, wconv1, bconv1 = self.conv1.backprop(y)

        grads = {
            'wlay3': wlay3, 'blay3': blay3,
            'wlay2': wlay2, 'blay2': blay2,
            # 'wlay1': wlay1, 'blay1': blay1,
            # 'wconv3': wconv3, 'bconv3': bconv3,
            'wconv2': wconv2, 'bconv2': bconv2,
            'wconv1': wconv1, 'bconv1': bconv1
        }

        return grads

    # обучение
    def train(self, features_train, target_train, batch_size, lr, print_log_freq, epochs):
        # выбор метода оптимизации
        optimizer = AdamGD(lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, params=self.get_params())
        # optimizer = SGD(lr=lr, params=self.get_params())

        total_loss = []
        total_acc = []

        for epoch in range(epochs):
            train_loader = Loader.dataloader(X=features_train, BATCH_SIZE=batch_size, y=target_train)
            i = 0
            loss_log = []
            acc_log = []
            # загрузка батча в модель
            for X_batch, y_batch in train_loader:
                # фитфорвард
                y_pred = self.feedforward(X_batch)

                # интерпретация таргета для лосса
                targ = []
                for jdx in range(len(y_batch)):
                    _ = []
                    for idx in range(10):
                        if y_batch[jdx] == idx:
                            _.append(1)
                        else:
                            _.append(0)
                    targ.append(_)
                targ = np.array(targ)

                # вычисление лосса
                loss = self.loss(targ, y_pred)
                # бэкпроп лосса
                x = self.loss.backprop(targ, y_pred)
                # бэкпроп слоев
                grad = self.backprop(x)
                # обновление весов
                params = optimizer.update_params(grad)
                self.set_params(params)

                loss_log.append(float(loss.sum()))
                # вычисление accuracy
                acc_log.append(y_pred.argmax() == targ.argmax())
                i += 1

                # вывод лосса и accuracy
                if i % print_log_freq == 0 and i != 0:
                    loss_avg = sum(loss_log[-print_log_freq:]) / print_log_freq
                    acc_avg = sum(acc_log[-print_log_freq:]) / print_log_freq
                    print(f'Loss: {loss_avg:.5f}, 'f'Acc: {acc_avg:.4f}, Epoch: {epoch}')
                    total_acc.append(acc_avg)
                    total_loss.append(loss_avg)

        excel_loss = pd.Series(total_loss)
        excel_loss.to_excel(r"C:\Users\makso\Desktop\ФООСИИ\CNN\data\loss.xlsx", index=False)
        excel_acc = pd.Series(total_acc)
        excel_acc.to_excel(r"C:\Users\makso\Desktop\ФООСИИ\CNN\data\acc.xlsx", index=False)

    # предсказание
    def predict(self, features_test):
        test_loader = Loader.dataloader(features_test, 1)

        y_pred_all = []
        y_pred_all_thresh = pd.DataFrame({0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []})
        for X_batch in test_loader:
            y_pred = self.feedforward(X_batch)

            y_pred_all.append(float(y_pred.argmax()))
            # для AUC-ROC
            y_pred = y_pred.T
            y_pred_ = {0: float(y_pred[0]), 1: float(y_pred[1]), 2: float(y_pred[2]), 3: float(y_pred[3]),
                       4: float(y_pred[4]), 5: float(y_pred[5]), 6: float(y_pred[6]),
                       7: float(y_pred[7]), 8: float(y_pred[8]), 9: float(y_pred[9])}
            y_pred_all_thresh = y_pred_all_thresh._append(y_pred_, ignore_index=True)

        return y_pred_all, y_pred_all_thresh

    def get_params(self):
        params = {
            'wlay3': self.layers[2].weight, 'blay3': self.layers[2].bias,
            'wlay2': self.layers[2].weight, 'blay2': self.layers[2].bias,
            # 'wlay1': self.layers[2].weight, 'blay1': self.layers[2].bias,
            # 'wconv3': self.layers[2].weight, 'bconv3': self.layers[2].bias,
            'wconv2': self.layers[1].weight, 'bconv2': self.layers[1].bias,
            'wconv1': self.layers[0].weight, 'bconv1': self.layers[0].bias
        }
        return params

    def set_params(self, params):
        self.layers[4].weight = params['wlay3']
        self.layers[4].bias = params['blay3']

        self.layers[3].weight = params['wlay2']
        self.layers[3].bias = params['blay2']
        #
        # self.layers[2].weight = params['wlay1']
        # self.layers[2].bias = params['blay1']

        # self.layers[2].weight = params['wconv3']
        # self.layers[2].bias = params['bconv3']

        self.layers[1].weight = params['wconv2']
        self.layers[1].bias = params['bconv2']

        self.layers[0].weight = params['wconv1']
        self.layers[0].bias = params['bconv1']


class Loss:
    def __call__(self, target, predict):
        return -target * np.log(predict)

    def backprop(self, target, predict):
        return -(target / predict)


class AdamGD:
    def __init__(self, lr, beta1, beta2, epsilon, params):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params

        self.momentum = {}
        self.rmsprop = {}

        for key in self.params:
            self.momentum['vd' + key] = np.zeros(self.params[key].shape)
            self.rmsprop['sd' + key] = np.zeros(self.params[key].shape)

    def update_params(self, grads):
        for key in self.params:
            self.momentum['vd' + key] = (self.beta1 * self.momentum['vd' + key]) + (1 - self.beta1) * grads[key]
            self.rmsprop['sd' + key] = (self.beta2 * self.rmsprop['sd' + key]) + (1 - self.beta2) * (grads[key] ** 2)
            self.params[key] = self.params[key] - (self.lr * self.momentum['vd' + key]) / (
                        np.sqrt(self.rmsprop['sd' + key]) + self.epsilon)
        return self.params


class SGD:
    def __init__(self, lr, params):
        self.lr = lr
        self.params = params

    def update_params(self, grads):
        for key in self.params:
            self.params[key] -= grads[key] * self.lr
        return self.params
