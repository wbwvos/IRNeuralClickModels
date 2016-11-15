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
import pickle
# since we are using stateful rnn tsteps can be set to 1

tsteps = 1
batch_size = 64
epochs = 300
serp_len = 10
inputs = 10242
num_hidden = 256
output = 1
# number of elements ahead that are used to make the prediction
x_train_e = np.load('data.npy')
y_train_e = np.load('labels.npy')
#y_train_e = np.reshape(y_train_e, (batch_size, serp_len))

print(x_train_e.shape, y_train_e.shape)

print('Creating Model')
model = Sequential()
model.add(TimeDistributed(Dense(num_hidden), input_shape=(serp_len, inputs)))
model.add(LSTM(num_hidden, return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='rmsprop')
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
        model.fit(x_train_e,
                  y_train_e,
                  batch_size=4,
                  verbose=1,
                  nb_epoch=1,
                  shuffle=False)
        model.reset_states()
    trainEnd = time.time()
    print('Trained the model in', trainEnd - trainStart, 'seconds')
    print('Saving the model...')
    model.save_weights(weights_filename, True)


print('Predicting')
predicted = model.predict(x_train_e, batch_size=batch_size, verbose=True)
print(predicted)
