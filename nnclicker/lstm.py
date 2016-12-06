#!/usr/bin/env python
import numpy as np
import time
import os.path
import sys
import cPickle as pickle
import glob

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
        self.epochs = 100
        self.num_hidden = 256
        self.num_output = 1
        self.batch_size = 64
        self.train_size = 7500
        self.validation_size = 1000
        self.test_size = 2500
        self.model = None

        self.data = None
        self.data_size = None
        self.train_set = None
        self.validation_set = None
        self.test_set = None

        # check data folder
        if os.path.exists('../../../../data'):
            # on server
            self.data_dir = '../../../../data'
        elif os.path.exists('../data'):
            # on laptop
            self.data_dir = '../data'
        else:
            print 'Data dir not found at ../data or ../../../../data'

        self.data_file = self.data_dir + "/sparse_matrix_set1_train_0-8000.pickle"
        self.weights_file = "lstm_weights_epoch_%d_train_size_%d_set1.dat" % (
            self.epochs, self.train_size)
        if len(sys.argv) == 2:
            if sys.argv[1] == "2":
                self.input_size = 11266
                self.data_file = self.data_dir + "/sparse_matrix_set2_train_0-10000.pickle"
                self.weights_file = "lstm_weights_epoch_%d_train_size_%d_set2.dat" % (
                    self.epochs, self.train_size)
            elif sys.argv[1] == "3":
                self.input_size = 21506
                self.data_file = self.data_dir + "/sparse_matrix_set3_train_0-10000.pickle"
                self.weights_file = "lstm_weights_epoch_%d_train_size_%d_set3.dat" % (
                    self.epochs, self.train_size)

    def load_data(self):
        """
        Function that loads data from data pickle
        """
        with open(self.data_file, "rb") as f:
            self.data = pickle.load(f)
            # self.data_size = len(self.data)

    def get_data_sets(self):
        """
        Function that divides data into train and test sets
        """
        self.train_set = self.data[:self.train_size]
        # self.validation_set = self.data[self.train_size:(self.train_size+self.validation_size)]
        self.test_set = self.data[-self.test_size:]
        del self.data

    def create_model(self):
        """
        Function that creates LSTM model
        """
        print "Creating Model"
        self.model = Sequential()
        self.model.add(LSTM(self.num_hidden, input_shape=(self.serp_len, self.input_size), return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.num_output, activation="sigmoid")))
        optim = Adadelta(rho=0.95, epsilon=1e-06, clipvalue=1.)
        self.model.compile(optimizer=optim, loss="binary_crossentropy", metrics=["accuracy"])
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

    def get_batch_pickle(self, batch_dir='/batches/'):
        """
        Function that reads train batch of given size
        """
        files = glob.glob(self.data_dir + batch_dir + '*.pickle')
        for fname in files:
            with open(self.data_dir + batch_dir + fname, 'rb') as f:
                batch_data = pickle.load(f)
                batch_X = np.zeros((len(batch_data), self.serp_len, self.input_size))
                batch_y = np.zeros((len(batch_data), self.serp_len, self.num_output))
                for (i, j) in enumerate(range(len(batch_data))):
                    matrix = batch_data[j].todense()
                    batch_X[i, :, 1:] = matrix[:, :-1]
                    batch_y[i, :] = matrix[:, -1]
                yield batch_X, batch_y
                del matrix, batch_X, batch_y, batch_data

    def get_train(self):
        train_X = np.zeros((self.train_size, self.serp_len, self.input_size))
        train_y = np.zeros((self.train_size, self.serp_len, self.num_output))
        for (i, sparse) in enumerate(self.train_set):
            matrix = sparse.todense()
            train_X[i, :, 1:] = matrix[:, :-1]
            train_y[i, :] = matrix[:, -1]
        return train_X, train_y

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
            self.model.load_weights(self.weights_file)
        else:
            print "Error: did not find weights file!"

    def train_lstm(self):
        """
        Function that trains LSTM model
        """
        print "Training the model..."
        train_starttime = time.time()
        train_X, train_y = self.get_train()
        self.model.fit(train_X, train_y, batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1, shuffle=True)
        print "Trained LSTM model in: %f seconds" % (time.time() - train_starttime)
        print('Saving the model...')
        self.model.save_weights(self.weights_file, True)

    def evaluate_lstm(self):
        """
        Function that evaluates lstm
        """
        print "Evaluating the model..."
        test_X, test_y = self.get_test()
        predict_probs = self.model.predict_proba(test_X)
        loglike_eval = evaluate.LogLikelihood()
        loglikelihood = loglike_eval.evaluate(predict_probs, test_y)
        print "LogLikelihood: %f" % loglikelihood

        # print "Examples:"
        # for i in range(10):
        #    print "%d------------------------------------" % i
        #    print "Predicted: %s" % str(predict_probs[i])
        #    print "Ground truth: %s" % str(test_y[i])
        #    print "======================================"


if __name__ == "__main__":
    start_time = time.time()
    lstmnn = LSTMNN()
    lstmnn.load_data()
    lstmnn.get_data_sets()
    lstmnn.create_model()
    lstmnn.train_lstm()
    lstmnn.evaluate_lstm()
