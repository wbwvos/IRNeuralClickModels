import numpy as np
import cPickle

batch_size = 64
serp_len = 10 # SERP contains 10 documents


# function that creates data example
def create_example(q, prev_i):
    # create document vector d
    d = np.round(np.maximum(0, 2.*np.random.randn(10240)))
    # create interaction vector i
    i = np.random.choice(2, p=(0.75, 0.25))
    return np.append(np.append(q, d), np.append(prev_i, i)).T


def create_serp(serp_len):
    q = 0
    i = 0
    serp = np.zeros((serp_len, 10243))
    for j in range(serp_len):
        example = create_example(q, i)
        i = example[-1]
        serp[j, :] = example
    return serp


def create_batch(batch_size, serp_len):
    batch = np.zeros((batch_size, serp_len, 10243))
    for j in range(batch_size):
        serp = create_serp(serp_len)
        batch[j, :, :] = serp
    return batch

batch = create_batch(batch_size, serp_len)
np.save('batch.npy', batch)
