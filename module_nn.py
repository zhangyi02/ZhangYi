import numpy as np


class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.W = np.random.random((output_size, input_size))
        self.b = np.zeros((output_size,1))
        self.output = np.zeros((output_size,1))

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activator.forward\
            (np.dot(self.W, input_data) + self.b)

    def backward(self, delta_data):
        self.delta = self.activator.backward(self.input) \
                     * np.dot(self.W.T, delta_data)
        self.W_grad = np.dot(delta_data, self.input.T)
        self.b_grad = delta_data

    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b ++ learning_rate * self.b_grad


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1/(1+np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


class Network(object):
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(
                layers[i], layers[i+1], SigmoidActivator()))

    def predict(self, last_output):
        for layer in self.layers:
            layer.forward(last_output)
            output = layer.output
        return output

    def calc_gradient(self, y):
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (y - self.layers[-1].output)

    def loss_Function(self, x ,y ,model, hprevs):


    def train(self, input, output, learning_rate, epochs):
        for epoch in range(epochs):
            self.predict =

