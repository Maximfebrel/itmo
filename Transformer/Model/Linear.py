import numpy as np
from Model.Optimizer import AdamGD, SGD


class FeedForward:
    def __init__(self, embed_dim, input_dim,  d_ff, learning_rate):
        self.W1 = np.random.rand(input_dim, d_ff) * 0.01
        self.W2 = np.random.rand(d_ff, input_dim) * 0.01

        self.b1 = np.random.rand(d_ff)
        self.b2 = np.random.rand(input_dim)

        self.x = None
        self.x1 = None
        self.learning_rate = learning_rate

    def feedforward(self, x):
        self.x = x
        self.x1 = self.sigmoid(np.dot(x, self.W1) + self.b1)
        x2 = self.sigmoid(np.dot(self.x1, self.W2) + self.b2)
        return x2

    def backward(self, d_output):
        params = {'w1': self.W1, 'b1': self.b1,
                  'w2': self.W2, 'b2': self.b2
                  }

        # optimizer = AdamGD(lr=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, params=params)
        optimizer = SGD(lr=self.learning_rate, params=params)

        dx2 = np.dot(d_output, self.W2.T) * self.dsigmoid(self.x1)
        dW2 = np.dot(self.x1.T, d_output)

        dx1 = np.dot(dx2, self.W1.T) * self.dsigmoid(self.x)
        dW1 = np.dot(self.x.T, dx2)

        grad = {
            'w1': dW1, 'b1': np.sum(dx2, axis=0, keepdims=True)[0],
            'w2': dW2, 'b2': np.sum(d_output, axis=0, keepdims=True)[0],
        }

        params = optimizer.update_params(grad)
        self.set_params(params)

        return dx1

    def set_params(self, params):
        self.W1 = params['w1']
        self.b1 = params['b1']
        self.W2 = params['w2']
        self.b2 = params['b2']

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def drelu(x):
        return (x > 0).astype(np.float32)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)




class MLP:
    def __init__(self, input_dim, learning_rate):
        self.W = np.random.rand(input_dim, 1) * 0.01
        self.b = np.random.rand(1)
        self.learning_rate = learning_rate

        self.x = None

    def feedforward(self, x):
        self.x = x
        return self.sigmoid(np.dot(x, self.W) + self.b)

    def backward(self, d_output):
        params = {'w': self.W, 'b': self.b}

        optimizer = AdamGD(lr=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, params=params)
        # optimizer = SGD(lr=self.learning_rate, params=params)

        dW = np.dot(self.x.T, d_output)
        dx = np.dot(d_output, self.W.T) * self.dsigmoid(self.x)

        grad = {
            'w': dW,
            'b': np.sum(d_output, axis=0, keepdims=True)[0]
        }

        params = optimizer.update_params(grad)
        self.set_params(params)

        return dx

    def set_params(self, params):
        self.W = params['w']
        self.b = params['b']

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)
