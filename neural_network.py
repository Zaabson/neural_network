import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.e ** (-x))

def identity_f(x):
    return x

def elu(x):
    alfa = 1
    if x <= 0:
        return alfa * ((np.e ** x) - 1)
    elif x > 0:
        return x

def sigmoid_deriv(x, y):
    return y*(1-y)


def identity_deriv(x, y):
    return np.ones(y.shape)

def tanh_deriv(x, y):
    return 1 - y ** 2

def elu_deriv(x, y):
    alfa = 1
    if x <= 0:
        return y + alfa
    elif x < 0:
        return 1


# the first dict contains all activation functions, the second contains their derivatives.
# Derivatives take as arguments both function argument and value, to be able to express them in the easiest form,
# for example sigmoid's derivative can be y(1-y)
activ_f_dict = {"sigmoid": sigmoid, "none": identity_f, "tanh": math.tanh, "elu": elu}
activ_deriv_dict = {"sigmoid": sigmoid_deriv, "none": identity_deriv, "tanh": tanh_deriv, "elu": elu_deriv}


class ConvolutionalNeuralNetwork:


    def __init__(self, dims, activations):
        """dims is a tuple with numbers of neurons in each layer
           activations is a tuple of names(strings) of activation functions for each layer
           ^ both start with the input layer"""

        self.n = len(dims)

        self.A = [np.zeros(x) for x in dims]
        self.Z = [np.zeros(x) for x in dims]  # first element in Z_layers is only to match index of A_layers

        self.b = np.array([[1]])
        self.activs = activations

        # initializing weights
        self.W = [None for _ in range(self.n - 1)]
        for x in range(self.n - 1):
            m, n = dims[x + 1], dims[x] + 1
            array = np.random.normal(0, 1, size=(m, n))
            self.W[x] = np.array(array)
        self.W = np.array(self.W)


    def forward_prop(self, input, output):

        self.Output = output.reshape((output.shape[0], 1))

        self.A[0] = np.append(input, self.b)
        self.A[0] = self.A[0].reshape((self.A[0].shape[0], 1))

        for x in range(self.n - 2):
            self.Z[x+1] = self.W[x] @ self.A[x]
            self.A[x+1] = activ_f_dict[self.activs[x+1]](self.Z[x+1])
            self.A[x+1] = np.concatenate((self.A[x+1], self.b))

        self.Z[-1] = self.W[-1] @ self.A[-2]
        self.A[-1] = activ_f_dict[self.activs[-1]](self.Z[-1])


    def backprop(self):

        self.dx = ["xd" for _ in range(self.n-1)]

        dcdz = 2 * (self.A[-1] - self.Output) * activ_deriv_dict[self.activs[-1]](self.Z[-1], self.A[-1])

        for x in range(1, self.n):

            self.dx[-x] = self.A[-x-1].T * dcdz

            dcdz = (self.W[-x].T[:-1] @ dcdz) * (activ_deriv_dict[self.activs[-x-1]](self.Z[-x-1], self.A[-x-1]))[:-1]

        self.dx = np.array(self.dx)


    def learn(self, inputs, outputs, epochs, batch_size=32, lr=0.001, mu=0.0):

        """classic SGD with mu = 0, otherwise uses nesterov momentum"""

        input_batches = np.array_split(inputs, batch_size)
        output_batches = np.array_split(outputs, batch_size)

        v = 0

        for i in range(epochs):

            for batch_ins, batch_outs in zip(input_batches, output_batches):

                y = self.W
                self.W = self.W + mu * v

                total_dx = np.zeros(self.W.shape)

                for input, output in zip(batch_ins, batch_outs):
                    self.forward_prop(input, output)

                    self.backprop()
                    total_dx = total_dx + self.dx

                v = mu * v - lr * total_dx
                self.W = y + v


    def learn_RMSprop(self, inputs, outputs, epochs, batch_size=32, lr=0.001, gamma=0.9):
        """RMSprop learning algorithm"""

        input_batches = np.array_split(inputs, batch_size)
        output_batches = np.array_split(outputs, batch_size)

        # msg is moving medium squared gradient, its biased towards 0
        msg = 0

        for i in range(epochs):

            for batch_ins, batch_outs in zip(input_batches, output_batches):

                total_dx = np.zeros(self.W.shape)

                for input, output in zip(batch_ins, batch_outs):
                    self.forward_prop(input, output)

                    self.backprop()
                    total_dx = total_dx + self.dx

                msg = gamma * msg + (1-gamma) * (total_dx ** 2)
                self.W = self.W - lr * total_dx / (msg ** 0.5 + 0.00001)


    def cost(self):

        c = (self.A[-1] - self.Output) ** 2
        return np.sum(c)

    def accuracy(self):

        if np.argmax(self.A[-1]) == np.argmax(self.Output):
            return 1
        else:
            return 0

    def evaluate(self, inputs, outputs):
        N = len(inputs)
        total_accuracy_score = 0
        total_cost = 0
        for input, output in zip(inputs, outputs):
            self.forward_prop(input, output)
            total_cost += self.cost()
            total_accuracy_score += self.accuracy()

        accuracy = total_accuracy_score / N
        avg_cost = total_cost / N
        print(f"average cost is: {avg_cost}, accuracy is: {accuracy}")
