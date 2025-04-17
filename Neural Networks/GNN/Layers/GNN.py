import numpy as np
from Layers.Optimizer import AdamGD


class GNN:
    def __init__(self, input_dim, hidden_dim, activation, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.activation = activation
        self.optimizer = AdamGD(lr=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)

        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, 10) * 0.01
        self.b2 = np.random.randn(10) * 0.01

        self.H = None
        self.A = None
        self.X = None
        self.AX = None

    def __call__(self, *args, **kwargs):
        self.A = args[0]
        self.X = args[1]

        return self.forward()

    def forward(self):
        self.message_passing()
        output = self.H @ self.W2 + self.b2
        return output

    def message_passing(self):
        self.AX = self.A @ self.X  # размерность x (размер батча, количество вершин в графе, количество признаков)
        self.AX = self.AX.reshape(self.AX.shape[0]*self.AX.shape[1], self.AX.shape[2])
        self.activation.input = self.AX
        self.H = self.activation(self.AX @ self.W1)

    def backward(self, d_out):
        params = {'w1': self.W1,
                  'w2': self.W2, 'b2': self.b2}

        self.optimizer(params)

        dL_dW2 = self.H.T @ d_out
        dL_dH = d_out @ self.W2.T

        dL_dW1 = self.AX.T @ dL_dH

        grad = {
            'w1': dL_dW1,
            'w2': dL_dW2,
            'b2': np.sum(dL_dH, axis=0, keepdims=True)[0]
        }

        params = self.optimizer.update_params(grad)
        self.set_params(params)

        if dL_dW1.shape[0] != self.input_dim:
            dx = self.activation.backward(dL_dH @ self.W1.T)
            return dx

    def set_params(self, params):
        self.W1 = params['w1']
        self.W2 = params['w2']
        self.b2 = params['b2']
