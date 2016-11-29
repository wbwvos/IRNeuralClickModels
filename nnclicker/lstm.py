#!/usr/bin/env python
import numpy as np
import time
import os.path
import cPickle as pickle

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from keras.optimizers import Adadelta

import evaluate

__author__ = 'Wolf Vos, Casper Thuis, Alexander van Someren, Jeroen Rooijmans'


class LSTMNN(object):
    """
    A LSTM Neural Network to train our click model
    """

    def __init__(self):
        self.tsteps = 1
        self.serp_len = 10
        self.input_size = 10242
        self.epochs = 500
        self.num_hidden = 256
        self.num_output = 1
        self.batch_size = 64
        self.train_size = 25000
        self.validation_size = 1000
        self.test_size = 25000
        self.model = None

        self.data = None
        self.data_size = None
        self.train_set = None
        self.validation_set = None
        self.test_set = None

        self.data_file =  "../../../../data/sparse_matrix_set1_train_0-651728.pickle"
        self.weights_file = "lstm_weights_epoch_%d_train_size_%d.dat" % (
            self.epochs, self.train_size)

    def load_data(self):
        """
        Function that loads data from data pickle
        """
        with open(self.data_file, "rb") as f:
            self.data = pickle.load(f)
        #self.data_size = len(self.data)

    def get_data_sets(self):
        """
        Function that divides data into train and test sets
        """
        self.train_set = self.data[:-(self.validation_size+self.test_size)]
        self.validation_set = self.data[-(self.validation_size+self.test_size):
                                        -self.test_size]
        self.test_set = self.data[-self.test_size:]
        del self.data

    def create_model(self):
        """
        Function that creates LSTM model
        """
        print "Creating Model"
        self.model = Sequential()
        self.model.add(TimeDistributed(Dense(self.num_hidden),
                                       input_shape=(self.serp_len, self.input_size)))
        self.model.add(LSTM(self.num_hidden, return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.num_output, activation="sigmoid")))
        optim = Adadelta(rho=0.95, epsilon=1e-06, clipvalue=1.)
        self.model.compile(optimizer=optim, loss="binary_crossentropy",
                           metrics=["accuracy"])
        self.model.summary()


    def get_batch_train(self):
        """
        Function that creates random train batch of given size
        """
        batch_X = np.zeros((self.train_size, self.serp_len, self.input_size))
        batch_y = np.zeros((self.train_size, self.serp_len, self.num_output))
        for (i, j) in enumerate(np.random.choice(len(self.train_set),
                                                     self.train_size, replace=False)):
            matrix = self.train_set[j].todense()
            batch_X[i, :, 1:] = matrix[:, :-1]
            batch_y[i, :] = matrix[:, -1]
        return batch_X, batch_y

    def get_val(self):
        """
        Function that creates validation set from sparse matrixes
        """
        val_X = np.zeros((self.validation_size, self.serp_len, self.input_size))
        val_y = np.zeros((self.validation_size, self.serp_len, self.num_output))
        for (i, sparse) in enumerate(self.validation_set):
            matrix = sparse.todense()
            val_X[i, :, 1:] = matrix[:, :-1]
            val_y[i, :] = matrix[:, -1]
        return val_X, val_y

    def get_test(self):
        """
        Funciton that creates test set from sparse matrixes
        """
        test_X = np.zeros((self.test_size, self.serp_len, self.input_size))
        test_y = np.zeros((self.test_size, self.serp_len, self.num_output))
        for (i, sparse) in enumerate(self.test_set):
            matrix = sparse.todense()
            test_X[i, :, 1:] = matrix[:, :-1]
            test_y[i, :] = matrix[:, -1]
        return test_X, test_y
    
    def load_lstm(self):
        """
        Function that loads LSTM model when weights are already saved
        """
        if os.path.isfile(self.weights_file):
            print "Found weights file, loading weights..."
            self.model.load_weights(self.weight_file)
        else:
            print "Error: did not find weights file!"


    def train_lstm(self):
        """
        Function that trains LSTM model
        """
        print "Training the model..."
        train_starttime = time.time()
        for i in range(self.epochs):
            print "Epoch: %d/%d" % (i+1, self.epochs)
            train_X, train_y = self.get_batch_train()
            val_X, val_y = self.get_val()
            self.model.fit(train_X, train_y,
                  validation_data=(val_X, val_y),
                  batch_size=self.batch_size,
                  nb_epoch=1,
                  verbose=1,
                  shuffle=True)
        print "Trained LSTM model in: %f seconds" % (time.time() - train_starttime)
        print('Saving the model...')
        self.model.save_weights(self.weights_file, True)

    def evaluate_lstm(self):
        """
        Function that evaluates lstm
        """
        #evaluation = evaluate.Perplexity()
        #test_X, test_y = self.get_test()
        #pred_probs = self.model.predict_proba(test_X)
        #perplexity, perplexity_rank = evaluation.evaluate(pred_probs, test_y)
        #print pred_probs[0]
        #print "Perplexity: %f" % perplexity
        #print "Perplexity@rank: %s" % str(perplexity_rank)

        print "Evaluating the model..."
        test_X, test_y = self.get_test()
        score = self.model.evaluate(test_X, test_y, verbose=1)
        print "Log likelihood: %.3f" % np.log(score[0])
        print "Accuracy: %.3f" % score[1]
        predict_probs = self.model.predict_proba(test_X)
        print "Examples:"
        for i in range(10):
            print "%d------------------------------------" % i
            print "Predicted: %s" % str(predict_probs[i])
            print "Ground truth: %s" % str(test_y[i])
            print "======================================"

if __name__ == "__main__":
    start_time = time.time()
    lstmnn = LSTMNN()
    lstmnn.load_data()
    lstmnn.get_data_sets()
    lstmnn.create_model()
    lstmnn.train_lstm()
    lstmnn.evaluate_lstm()
