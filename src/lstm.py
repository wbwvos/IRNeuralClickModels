'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
import time
import os.path
#import pickle
import cPickle as pickle
from scipy.sparse import coo_matrix

# since we are using stateful rnn tsteps can be set to 1

tsteps = 1
batch_size = 64
epochs = 200
serp_len = 10
inputs = 10242
num_hidden = 256
output = 1
# number of elements ahead that are used to make the prediction
#x_train = np.load('data.npy')
#y_train = np.load('labels.npy')

with open('data_list.cpickle', 'rb') as f:
    data = pickle.load(f)

x_data = np.zeros((395, 10, 10242))
y_data = np.zeros((395, 10, 1))
for i, d in enumerate(data):
    matrix = d.todense()
    x_data[i, :, 1:] = matrix[:, :-1]
    y_data[i, :] = matrix[:, -1]

x_train = x_data[:300, :, :]
y_train = y_data[:300, :, :]
x_val = x_data[300:, :, :]
y_val = y_data[300:, :, :]

print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_val:', x_val.shape)
print('y_val:', y_val.shape)

print('Creating Model')
model = Sequential()
model.add(TimeDistributed(Dense(num_hidden), input_shape=(serp_len, inputs)))
model.add(LSTM(num_hidden, return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
weights_filename = 'weights.dat'

if os.path.isfile(weights_filename):
    print('Loading the model...')
    model.load_weights(weights_filename)
else:
    print('Training the model...')
    trainStart = time.time()
    for i in range(epochs):
        print('Epoch', i+1, '/', epochs)
        model.fit(x_train,
                  y_train,
                  batch_size=64,
                  verbose=1,
                  nb_epoch=1,
                  shuffle=True)
        #model.reset_states()
    trainEnd = time.time()
    print('Trained the model in', trainEnd - trainStart, 'seconds')
    print('Saving the model...')
    model.save_weights(weights_filename, True)


print('Evaluating')
score = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=1)
print('log likelihood:  ', score[0])
print('Accuracy:        ', score[1])
