import numpy as np
from Model.Linear import MLP, FeedForward
from Model.Optimizer import AdamGD, SGD
from Model.ButchNorm import LayerNorm


class MultiHeadAttention:
    def __init__(self, input_dim, embed_dim, num_heads, learning_rate):
        self.num_heads = num_heads
        self.depth = embed_dim // num_heads

        self.Wq = np.random.rand(input_dim, embed_dim) * 0.01
        self.Wk = np.random.rand(input_dim, embed_dim) * 0.01
        self.Wv = np.random.rand(input_dim, embed_dim) * 0.01
        self.Wo = np.random.rand(embed_dim, input_dim) * 0.01

        self.learning_rate = learning_rate
        self.x = None
        self.attention = None

    def split_heads(self, x):
        batch_size = x.shape[0]
        x = np.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def scaled_attention(self, q, k, v):
        dk = q.shape[-1]
        scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) / np.sqrt(dk)

        attention_weights = self.softmax(scores)
        output = np.matmul(attention_weights, v)
        return output

    def feedforward(self, v, k, q):
        self.x = q
        q = np.dot(q, self.Wq)
        k = np.dot(k, self.Wk)
        v = np.dot(v, self.Wv)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attention = self.scaled_attention(q, k, v)
        self.attention = np.reshape(attention, (attention.shape[0], self.num_heads * self.depth))
        o = np.dot(self.attention, self.Wo)
        return o

    def backward(self, d_output):
        params = {
            'wq': self.Wq,
            'wk': self.Wk,
            'wv': self.Wv,
            'wo': self.Wo
            }

        # optimizer = AdamGD(lr=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, params=params)
        optimizer = SGD(lr=self.learning_rate, params=params)

        do = np.dot(d_output, self.Wo.T)
        d_Wo = np.dot(self.attention.T, d_output)

        d_q = np.dot(do, self.Wq.T)

        d_Wq = np.dot(self.x.T, do)
        d_Wk = np.dot(self.x.T, do)
        d_Wv = np.dot(self.x.T, do)

        grad = {
            'wq': d_Wq,
            'wk': d_Wk,
            'wv': d_Wv,
            'wo': d_Wo
        }

        params = optimizer.update_params(grad)
        self.set_params(params)

        return d_q

    def set_params(self, params):
        self.Wq = params['wq']
        self.Wk = params['wk']
        self.Wv = params['wv']
        self.Wo = params['wo']


class EncoderLayer:
    def __init__(self, input_dim, embed_dim, num_heads, d_ff, learning_rate):
        self.mha = MultiHeadAttention(input_dim, embed_dim, num_heads, learning_rate)
        self.ffn = FeedForward(embed_dim, input_dim, d_ff, learning_rate)
        self.layernorm1 = LayerNorm(1, 1, 0.001)
        self.layernorm2 = LayerNorm(1, 1, 0.001)

    def feedforward(self, x):
        attn_output = self.mha.feedforward(x, x, x)
        out1 = self.layernorm1.feedforward(attn_output+x)
        ffn_output = self.ffn.feedforward(out1)
        out2 = self.layernorm2.feedforward(ffn_output+out1)
        return out2

    def backward(self, d_output):
        dout2 = self.layernorm2.backward(d_output)
        d_ff_out = self.ffn.backward(dout2)
        dout1 = self.layernorm1.backward(d_ff_out+d_output)
        d_attn_out = self.mha.backward(dout1)
        return d_attn_out+dout1


class Encoder:
    def __init__(self, num_layers, input_dim, embed_dim, num_heads, d_ff, learning_rate):
        self.enc_layers = [EncoderLayer(input_dim, embed_dim, num_heads, d_ff, learning_rate) for _ in range(num_layers)]

    def feedforward(self, x):
        for enc_layer in self.enc_layers:
            x = enc_layer.feedforward(x)
        return x


class PositionalEncoding:
    def __init__(self, embed_dim, max_len):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        self.angle_rads = pos * angle_rates
        self.angle_rads[:, 0::2] = np.sin(self.angle_rads[:, 0::2])
        self.angle_rads[:, 1::2] = np.cos(self.angle_rads[:, 1::2])

    def feedforward(self, x):
        seq_len = x.shape[1]
        return x.toarray() + self.angle_rads[:seq_len, :]


class Transformer:
    def __init__(self, num_layers, input_dim, embed_dim, num_heads, ff_dim, butch_size, epochs, learning_rate):
        self.encoder = Encoder(num_layers, input_dim, embed_dim, num_heads, ff_dim, learning_rate)
        self.positional_encoding = PositionalEncoding(input_dim, butch_size)
        self.mlp = MLP(input_dim, learning_rate)

        self.butch_size = butch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self, X, y):
        loss = []

        for epoch in range(self.epochs):
            pred = []
            for i in range(0, X.shape[0], self.butch_size):
                X_butch = X[i: i+self.butch_size]
                y_butch = y[i: i+self.butch_size]

                enc_input = self.positional_encoding.feedforward(X_butch)
                enc_output = self.encoder.feedforward(np.array(enc_input))
                out = self.mlp.feedforward(enc_output)

                for j in list(out):
                    if float(j) > 0.5:
                        pred.append(1)
                    else:
                        pred.append(0)

                self.backward(y_butch, out)

            print(self.binary_cross_entropy(y, pred))
            loss.append(self.binary_cross_entropy(y, pred))

        return loss

    def predict(self, X):
        pred = []
        out_list = []
        for i in range(0, X.shape[0], self.butch_size):
            X_butch = X[i: i + self.butch_size]

            enc_input = self.positional_encoding.feedforward(X_butch)
            enc_output = self.encoder.feedforward(np.array(enc_input))
            out = self.mlp.feedforward(enc_output)

            out_list.extend(list(out))
            for j in list(out):
                if float(j) > 0.5:
                    pred.append(1)
                else:
                    pred.append(0)
        return pred, out_list

    def backward(self, y_true, y_pred):
        d_loss = (y_pred - np.array([y_true]).T) / y_pred.shape[0]
        d_out = self.mlp.backward(d_loss)

        # Обратный проход через энкодер
        for layer in reversed(self.encoder.enc_layers):
            d_out = layer.backward(d_out)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        return -np.mean(y_true * np.log(np.array(y_pred) + 1e-9)) / 10
