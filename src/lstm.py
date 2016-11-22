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




tsteps = 1
batch_size = 16
epochs = 50
serp_len = 10
inputs = 10242
num_hidden = 256
output = 1
train_batch_size = 128
val_set_size = 32

with open('data_list.cpickle', 'rb') as f:
    data = pickle.load(f)

def next_batch(data, train_batch_size, val_set_size):
    x_data = np.zeros((train_batch_size, 10, 10242))
    y_data = np.zeros((train_batch_size, 10, 1))
    for i, j in enumerate(np.random.choice(len(data)-(val_set_size*2), train_batch_size, replace=False)):
        matrix = data[j].todense()
        x_data[i, :, 1:] = matrix[:, :-1]
        y_data[i, :] = matrix[:, -1]
    return x_data, y_data

def val_set(data, val_set_size):
    x_data = np.zeros((val_set_size, 10, 10242))
    y_data = np.zeros((val_set_size, 10, 1))
    for i, j in enumerate(xrange(len(data)-(val_set_size*2), len(data)-val_set_size)):
        matrix = data[j].todense()
        x_data[i, :, 1:] = matrix[:, :-1]
        y_data[i, :] = matrix[:, -1]
    return x_data, y_data

def test_set(data, val_set_size):
    x_data = np.zeros((val_set_size, 10, 10242))
    y_data = np.zeros((val_set_size, 10, 1))
    for i, j in enumerate(xrange(len(data)-val_set_size, len(data))):
        matrix = data[j].todense()
        x_data[i, :, 1:] = matrix[:, :-1]
        y_data[i, :] = matrix[:, -1]
    return x_data, y_data

x_train, y_train = next_batch(data, train_batch_size, val_set_size)
x_val, y_val = val_set(data, val_set_size)
x_test, y_test = test_set(data, val_set_size)

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
        x_train_batch, y_train_batch = next_batch(data, train_batch_size, val_set_size)
        model.fit(x_train_batch,
                  y_train_batch,
                  validation_data=(x_val, y_val),
                  batch_size=batch_size,
                  verbose=1,
                  nb_epoch=1,
                  shuffle=True)
        #model.reset_states()
    trainEnd = time.time()
    print('Trained the model in', trainEnd - trainStart, 'seconds')
    print('Saving the model...')
    model.save_weights(weights_filename, True)


print('Evaluating')
score = model.evaluate(x_test, y_test, verbose=1)
print('log likelihood:  ', score[0])
print('Accuracy:        ', score[1])

print('Predict example')
predict = model.predict_classes(x_test)
print(predict[8])

print(np.sum(predict))
print(predict.shape)
print(np.sum(y_test))
print(y_test.shape)

#def perplexity(y_true, y_pred, mask=None):
#    if mask is not None:
#        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#        mask = K.permute_dimensions(K.reshape(mask, y_true.shape[:-1]), (0, 1, 'x'))
#        truth_mask = K.flatten(y_true*mask).nonzero()[0]  ### How do you do this on tensorflow?
#        predictions = K.gather(y_pred.flatten(), truth_mask)
#        return K.pow(2, K.mean(-K.log2(predictions)))
#    else:
#        return K
