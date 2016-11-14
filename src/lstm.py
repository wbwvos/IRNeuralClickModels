'''
TOY DATA LSTM
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

# workaround control_flow_ops
import tensorflow as tf
tf.python.control_flow_ops = tf

# fix random seed for reproducibility
np.random.seed(7)


def load_data():
    # import dataset
    data = np.load('batch.npy')
    return (data[:, :, :-1], data[:, :, -1])

# load data
(train_X, train_y) = load_data()
print('Loaded data!')
print('Train data shape: ({0},{1})').format(str(train_X.shape),
                                            str(train_y.shape))

# DEFINE LSTM MODEL #
# create model
model = Sequential()
model.add(LSTM(256, batch_input_shape=(1, 10, 10242)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# show model
print model.summary()

# train model
model.fit(train_X, train_y, nb_epoch=5, batch_size=64)
