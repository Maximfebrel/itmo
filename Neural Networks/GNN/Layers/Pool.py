import numpy as np


class AvgPool:
    def __init__(self, num_nodes, embed_size):
        self.x = None
        self.x_shape = None
        self.num_nodes = num_nodes
        self.embed_size = embed_size

    def __call__(self, *args, **kwargs):
        self.x = args[0]

        self.x_shape = self.x.shape
        self.x = self.x.reshape(self.x.shape[0] // self.num_nodes, self.num_nodes, self.embed_size)
        out = np.mean(self.x, axis=1)
        return out

    def backward(self, d_out):
        out = np.repeat(d_out, self.x.shape[1])
        out = out.reshape(self.x_shape)
        return out


class MaxPool:
    def __init__(self, num_nodes, embed_size):
        self.x = None
        self.x_shape = None
        self.num_nodes = num_nodes
        self.embed_size = embed_size

    def __call__(self, *args, **kwargs):
        self.x = args[0]

        self.x_shape = self.x.shape
        self.x = self.x.reshape(self.x.shape[0] // self.num_nodes, self.num_nodes, self.embed_size)
        out = np.max(self.x, axis=1)
        return out

    def backward(self, d_out):
        out = np.repeat(d_out, self.x.shape[1])
        out = out.reshape(self.x_shape)
        return out


class Flatten:
    def __init__(self, num_nodes, embed_size):
        self.x = None
        self.num_nodes = num_nodes
        self.embed_size = embed_size

    def __call__(self, *args, **kwargs):
        self.x = args[0]

        out = self.x.reshape(self.x.shape[0] // self.num_nodes, self.num_nodes*self.embed_size)
        return out

    def backward(self, d_out):
        #(3072*10)
        #(16*1920)
        d_out = d_out.reshape(self.x.shape)
        return d_out
