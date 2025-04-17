import numpy as np
from Layers.Optimizer import AdamGD


class Linear:
    def __init__(self, hidden_dim, out_dim, learning_rate=0.01, activation=None):
        self.W1 = np.random.rand(hidden_dim, out_dim) * 0.1
        self.b1 = np.random.rand(out_dim) * 0.1

        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = AdamGD(lr=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)

        self.x = None

    def __call__(self, *args, **kwargs):
        self.x = args[0]
        if self.activation is not None:
            self.activation.input = self.x
            out = self.activation(np.dot(self.x, self.W1) + self.b1)
        else:
            out = np.dot(self.x, self.W1) + self.b1
        return out

    def backward(self, d_output):
        params = {'w1': self.W1, 'b1': self.b1}

        self.optimizer(params)

        dW = np.dot(self.x.T, d_output)
        if self.activation is not None:
            dx = self.activation.backward(np.dot(d_output, self.W1.T))
        else:
            dx = np.dot(d_output, self.W1.T)

        grad = {'w1': dW, 'b1': np.sum(d_output, axis=0, keepdims=True)[0]}

        params = self.optimizer.update_params(grad)
        self.set_params(params)

        return dx

    def set_params(self, params):
        self.W1 = params['w1']
        self.b1 = params['b1']
