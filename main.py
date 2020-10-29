import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(o, 1) for o in sizes[1:]]
        self.weights = [np.random.randn(o, i) for o, i in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
            return a


    def sgd(self, training_data, epochs, batch_size, eta, test_data=None):
        global n_test
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            batches = [training_data[k: k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_network(batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}"
                      .format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_network(self, batch, eta):
        pass

    def evaluate(self, test_data):
        pass

