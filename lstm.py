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
    data = np.load('data.npy')
    split1 = np.array_split(data, 10)
    split2 = np.concatenate(split1[:9])
    test = split1[9]
    split3 = np.array_split(split2, 10)
    train = np.concatenate(split3[:9])
    val = split3[9]
    return (train[:, 0], train[:, 1]), (val[:, 0], val[:, 1]), (test[:, 0],
                                                                test[:, 1])

# load data
(train_X, train_y), (val_X, val_y), (test_X, test_y) = load_data()
print('Loaded data!')
print('Train data shape: {0}').format(str(train_X.shape))
print('Validation data shape: {0}').format(str(val_X.shape))
print('Test data shape: {0}').format(str(test_X.shape))

print train_X.shape, train_y.shape
print train_X, train_y
# DEFINE LSTM MODEL #
# create model
model = Sequential()
model.add(Embedding(10, 10, input_length=train_X.shape[0]))
model.add(LSTM(100), dropout_W=0.2, dropout_U=0.2)
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# show model
print model.summary()

# train model
model.fit(train_X, train_y, validation_data=(val_X, val_y), nb_epoch=1,
          batch_size=32)

# Final evaluation of the model
scores = model.evaluate(test_X, test_y, verbose=1)
print('Accuracy: {.2f}').format(scores[1]*100)
