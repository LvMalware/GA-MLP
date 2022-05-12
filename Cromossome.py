import numpy as np
from random import randint, random

def func(ax_b):
    return 1.0 / (1.0 + np.exp(-ax_b))

class Cromossome():
    def __init__(self, ninput, noutput, data, hlayers=(32,), weights=None, bias=None):
        self.data = data
        self.score = None
        self.ninput = ninput
        self.noutput = noutput

        if bias:
            self.bias = bias
        else:
            self.bias = [np.array([0] * n) for n in hlayers] #[ np.random.uniform(0, 1, size=(n,)) for n in hlayers ]
            self.bias.append(np.array([0] * noutput))

        if weights:
            self.weights = weights
            self.hlayers = (x.shape[2] for x in weights)
        else:
            self.hlayers = hlayers
            layers = [ ninput, *hlayers, noutput ]
            self.weights = [ np.random.uniform(-1, 1, size=(layers[i], layers[i + 1])) for i in range(len(layers) - 1) ]

    def fitness(self):

        if self.score is not None:
            return self.score
        pred = self.classify()
        targ = self.data[1]
        self.score = sum(sum((pred - targ) ** 2))
        return self.score

    def classify(self, data=None):
        x, y = data if data is not None else self.data
        for bias, layer in zip(self.bias, self.weights):
            x = func(x.dot(layer) + bias)
        return x

    def mutate(self):
        layer = randint(0, len(self.weights) - 1)
        #if randint(0, 2) != 0:
        change = np.random.uniform(-0.1, 0.1, size=self.weights[layer].shape)
        self.weights[layer] += change
        #else:
        #    size = len(self.bias[layer])
        #    for _ in range(randint(1, size)):
        #        self.bias[layer][randint(0, size - 1)] += random() - random()

    def __lt__(self, other):
        return self.fitness() < other.fitness()

    def __str__(self):
        return f"Cromossome(fitness={self.fitness()})"

    def __gt__(self, other):
        return self.fitness() > other.fitness()

    def __eq__(self, other):
        return self.fitness() == other.fitness()

    def crossover(self, other, mutation):
        if self < other:
            first, second = self, other
        else:
            first, second = other, self

        layer = randint(0, len(first.weights) - 1)
        w = [ np.array(x) for x in first.weights ]

        for _ in range(w[layer].shape[0]):
            i, j = randint(0, w[layer].shape[0] - 1), randint(0, w[layer].shape[1] - 1)
            w[layer][i][j] = second.weights[layer][i][j]

        a = Cromossome(self.ninput, self.noutput, self.data, self.hlayers, w, first.bias)
        b = Cromossome(self.ninput, self.noutput, self.data, self.hlayers, w, second.bias)

        if random() < mutation:
            a.mutate()
            b.mutate()

        return a if a < b else b
