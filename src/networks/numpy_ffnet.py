import numpy as np


class NumpyFfnet:
    ############################################################################################################
    def __init__(self, input_size, hidden_size, output_size, weight_stdev):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_stdev = weight_stdev

        self.h_bias = np.random.normal(0, self.weight_stdev, [self.hidden_size])
        self.h_x = np.random.normal(0, self.weight_stdev, [self.hidden_size, self.input_size])

        self.o_bias = np.random.normal(0, self.weight_stdev, [self.output_size])
        self.o_h = np.random.normal(0, self.weight_stdev, [self.output_size, self.hidden_size])

        np.set_printoptions(suppress=True, precision=3, floatmode='fixed', linewidth=np.inf)

    ############################################################################################################
    def train(self, x, y, learning_rate):
        h, o = self.feedforward(x)
        o_cost = self.calc_cost(y, o)
        self.backpropogation(x, o, h, o_cost, learning_rate)

    ############################################################################################################
    def test(self, x, y):
        h, o = self.feedforward(x)
        o_cost = self.calc_cost(y, o)
        return o, h, o_cost

    ############################################################################################################
    def feedforward(self, x):
        h = self.tanh(np.dot(self.h_x, x) + self.h_bias)
        o = self.sigmoid(np.dot(self.o_h, h) + self.o_bias)
        return h, o

    ############################################################################################################
    @staticmethod
    def calc_cost(y, o):
        return y - o

    ############################################################################################################
    def backpropogation(self, x, o, h, o_cost, learning_rate):
        o_delta = o_cost * self.sigmoid_prime(o)

        h_cost = np.dot(o_delta, self.o_h)
        h_delta = h_cost * self.tanh_prime(h)

        self.o_bias += o_delta * learning_rate
        self.o_h += (np.dot(o_delta.reshape(len(o_delta), 1), h.reshape(1, len(h))) * learning_rate)

        self.h_bias += h_delta * learning_rate
        self.h_x += (np.dot(h_delta.reshape(len(h_delta), 1), x.reshape(1, len(x))) * learning_rate)

    ############################################################################################################
    @staticmethod
    def tanh(z):
        return np.tanh(z)

    ############################################################################################################
    @staticmethod
    def tanh_prime(z):
        return 1.0 - np.tanh(z)**2

    ############################################################################################################
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    ############################################################################################################
    @staticmethod
    def sigmoid_prime(z):
        return 1/(1+np.exp(-z)) * (1 - 1/(1+np.exp(-z)))

    ############################################################################################################
    def save_weights(self, filename):
        f = open(filename, 'w')

        h_bias_string = "x0"
        for i in range(self.hidden_size):
            h_bias_string += ",{}".format(self.h_bias[i])
        f.write(h_bias_string+"\n")

        for i in range(self.input_size):
            h_weight_string = "x{}".format(i+1)
            weight = self.h_x[:, i]
            for j in range(self.hidden_size):
                h_weight_string += ",{}".format(weight[j])
            f.write(h_weight_string + "\n")

        o_bias_string = "h0"
        for i in range(self.output_size):
            o_bias_string += ",{}".format(self.o_bias[i])
        f.write(o_bias_string+"\n")

        for i in range(self.hidden_size):
            o_weight_string = "h{}".format(i+1)
            weight = self.o_h[:, i]
            for j in range(self.output_size):
                o_weight_string += ",{}".format(weight[j])
            f.write(o_weight_string + "\n")

        f.close()

    ############################################################################################################
    def load_weights(self, filename):
        f = open(filename)
        for line in f:
            data = (line.strip().strip('\n').strip()).split(',')
            label = data[0]
            weights = data[1:]

            if label[0] == 'x':
                if label[1] == '0':
                    for i in range(len(weights)):
                        self.h_bias[i] = float(weights[i])
                else:
                    column = int(label[1:]) - 1
                    for i in range(len(weights)):
                        self.h_x[i, column] = float(weights[i])

            elif label[0] == 'h':
                if label[1] == '0':
                    for i in range(len(weights)):
                        self.o_bias[i] = float(weights[i])
                else:
                    column = int(label[1:]) - 1
                    for i in range(len(weights)):
                        self.o_h[i, column] = float(weights[i])
        f.close()

