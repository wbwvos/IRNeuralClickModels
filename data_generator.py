import numpy as np
import cPickle

num_items = 200
data = []

for i in range(num_items):
    # create SERP
    serp = np.random.permutation(10)
    # create data vector of length 10240
    vec = np.random.choice(2, size=10, p=(0.9, 0.1))
    data.append((serp, vec))

np.save('data.npy', data)
