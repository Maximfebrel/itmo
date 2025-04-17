import numpy as np

from Layers.GNN import GNN
from Layers.Linear import Linear
from Layers.Activation import Loss
from Layers.Pool import AvgPool, MaxPool, Flatten
from Layers.Activation import Sigmoid, Relu, Tanh


class GraphRegressor:
    def __init__(self, input_dim, hidden_dim, num_nodes, learning_rate=0.01, butch_size=16, epochs=100):
        self.conv1 = GNN(input_dim, hidden_dim, Tanh(), learning_rate)
        self.pool = Flatten(num_nodes, hidden_dim)
        self.linear1 = Linear(hidden_dim*num_nodes, 64, learning_rate, Tanh())
        self.linear2 = Linear(64, 1, learning_rate)
        self.loss = Loss()

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.butch_size = butch_size

    def fit(self, A, X, y):
        loss = []
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.butch_size):
                A_batch = A[i:i + self.butch_size]
                X_batch = X[i:i + self.butch_size]
                y_batch = y[i:i + self.butch_size]

                output = self.forward(A_batch, X_batch)

                self.backward(y_batch, output)

            loss.append(self.compute_loss(A, X, y))
            print(f'Epoch {epoch}: Loss {loss[-1]}')
        return loss

    def forward(self, A_batch, X_batch):
        X_batch = self.conv1(A_batch, X_batch)

        X_batch = self.pool(X_batch)

        X_batch = self.linear1(X_batch)
        output = self.linear2(X_batch)

        return output

    def backward(self, y_batch, output):
        d_loss = self.loss.backward(output, y_batch)

        d_out = self.linear2.backward(d_loss)
        d_out = self.linear1.backward(d_out)

        d_out = self.pool.backward(d_out)

        self.conv1.backward(d_out)

    def compute_loss(self, A, X, y):
        out = [self.forward(A_.reshape(1, A_.shape[0], A_.shape[1]), X_.reshape(1, X_.shape[0], X_.shape[1])) for A_, X_ in zip(A, X)]
        out = np.array(out)
        out = out.reshape(out.shape[0], 1)
        return self.loss(y, out)

    def predict(self, A, X):
        out = [self.forward(A_.reshape(1, A_.shape[0], A_.shape[1]), X_.reshape(1, X_.shape[0], X_.shape[1])) for A_, X_ in zip(A, X)]
        out = np.array(out)
        out = out.reshape(out.shape[0], 1)
        return out
