#!/usr/bin/env python3

# Author: Lucas V. Araujo <lucas.vieira.ar@disroot.org>

import numpy as np

from GA import GA

ninput = 3
noutput = 1
hlayers = (32,)

n = 10
X = np.random.normal(0.1, 1.0, size=(n, ninput))
Y = np.random.choice((0.0, 1.0), size=(n, noutput))

data = [X, Y]

ga = GA((ninput, noutput, data, hlayers), 1000, 500, 150, 0.5, 0.8)
mlp = ga.evolve(200, 0.001)

pred = np.round(mlp.classify(data))
accuracy = 100 * sum([1 if a == b else 0 for a, b in zip(pred, Y)])/ len(Y)
print(f"Accuracy for trainning data: {accuracy}%")

