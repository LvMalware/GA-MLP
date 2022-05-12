import numpy as np

from GA import GA


ninput = 3
noutput = 1
hlayers = (10, 7)

n = 10
X = np.random.normal(0.1, 0.5, size=(n, ninput))
Y = np.abs(np.round(np.random.normal(0, 1, size=(n, noutput))))
data = [X, Y]

ga = GA((ninput, noutput, data, hlayers), 1000, 300, 100, 0.5, 0.8)
mlp = ga.evolve(200, 0.001)

print("Target:\n", Y)
print("Predict:\n", np.round(mlp.classify(data)))
