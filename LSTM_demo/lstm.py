'''
IMDB REVIEW SENTATMENT ANALYSIS LSTM DEMO
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(7)
# import dataset
from keras.datasets import imdb

# LOAD AND PREPROCESS DATA #####

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# DEFINE LSTM MODEL #####
# create the model
embedding_vector_length = 32
# add embedding layer that represent words in vector of 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length,
                    input_length=max_review_length))

# add lstm layer with 100 units with dropout
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))

# add dense output layer with 1 unit and sigmoid activation
model.add(Dense(1, activation='sigmoid'))
# use binary crossentropy as loglos function, adam optimizes parameters
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# show model
print model.summary()

# train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3,
          batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
