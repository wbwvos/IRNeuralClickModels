import numpy as np

batch_size = 64
serp_len = 10  # SERP contains 10 documents


# function that initiates vector values
def init_vectors():
    # query vector is zero vector of size 1
    q = np.ones(1)
    # user interaction at state 0 is empty
    i = np.ones(1, dtype='int')
    # document vector is empty before document
    # is examined
    d = np.zeros(10240)
    return (q, i, d)

# init data list
data = np.zeros((batch_size, serp_len, 10242))
labels = np.zeros((batch_size, serp_len, 1))

# init vectors (state0)
(q, i, d) = init_vectors()

# fill batch
for j in range(batch_size):
    # fill serp
    for k in range(serp_len):
        # create dummy document vector
        d = np.round(np.maximum(0, .5*np.random.randn(d.shape[0])))
        # create dummy interaction
        interaction = np.array([np.random.choice(2, p=(.75, .25))])
        # concat vectors
        state_vec = np.zeros(10242)
        state_vec[0] = q
        state_vec[1] = i
        state_vec[2:] = d
        # fill data
        data[j, k, :] = state_vec
        # fill label
        labels[j, k] = interaction
        # update interaction
        i = interaction

np.save('data.npy', data)
np.save('labels.npy', labels)
