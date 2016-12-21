#!/usr/bin/env python
import numpy as np
import time
import os.path
import sys
import cPickle as pickle
import progressbar

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
        self.epochs = 5
        self.num_hidden = 256
        self.num_output = 1
        self.batch_size = 64
        self.train_size = 500000
        self.validation_size = 1000
        self.test_size = 100000
        self.model = None
        self.data = None
        self.data_size = None
        self.train_set = None
        self.validation_set = None
        self.test_set = None
        self.print_step = 100
        self.train = False

        # check data folder
        if(os.path.exists('../../../../data')):
            # on server
            self.data_dir = '../../../../data'
        elif(os.path.exists('../data')):
            # on laptop
            self.data_dir = '../data'
        else:
            print 'Data dir not found at ../data or ../../../../data'

        self.train_dir = self.data_dir + "/sparse_matrix_set1_train_0-500000.pickle/"
        self.test_dir = self.data_dir + "/sparse_matrix_set1_train_500000-600000.pickle/"
        self.weights_file = "lstm_weights_epoch_%d_train_size_%d_set1.dat" % (
            self.epochs, self.train_size)

        if len(sys.argv) >= 2:
            if sys.argv[1] == "2":
                self.input_size = 11266
                self.train_dir = self.data_dir + "/sparse_matrix_set2_train_0-500000.pickle/"
                self.test_dir = self.data_dir + "/sparse_matrix_set2_train_500000-600000.pickle/"
                self.weights_file = "lstm_weights_epoch_%d_train_size_%d_set2.dat" % (
                    self.epochs, self.train_size)

            elif sys.argv[1] == "3":
                self.input_size = 21506
                self.train_dir = self.data_dir + "/sparse_matrix_set3_train_0-500000.pickle/"
                self.test_dir = self.data_dir + "/sparse_matrix_set3_train_500000-600000.pickle/"
                self.weights_file = "lstm_weights_epoch_%d_train_size_%d_set3.dat" % (
                    self.epochs, self.train_size)

            elif sys.argv[1] == "4":
                self.input_size = 22530
                self.train_dir = self.data_dir + "/sparse_matrix_set4_train_0-500000.pickle/"
                self.test_dir = self.data_dir + "/sparse_matrix_set4_train_500000-600000.pickle/"
                self.weights_file = "lstm_weights_epoch_%d_train_size_%d_set4.dat" % (
                    self.epochs, self.train_size)

        if len(sys.argv) == 3:
            if sys.argv[2] == "train":
                self.train = True
            elif sys.argv[2] == "eval":
                self.train = False
            else:
                print "pass argument \"train\" to train lstm or \"eval\" to evaluate lstm"

    def create_model(self):
        """
        Function that creates LSTM model
        """
        print "Creating Model"
        self.model = Sequential()
        self.model.add(LSTM(self.num_hidden, input_shape=(self.serp_len, self.input_size),
                            return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.num_output, activation="sigmoid")))
        optim = Adadelta(rho=0.95, epsilon=1e-06, clipvalue=1.)
        self.model.compile(optimizer=optim, loss="binary_crossentropy", metrics=["accuracy"])
        self.model.summary()

    def get_batch_pickle(self, data_dir):
        """
        Function that reads batches from given data_dir
        """
        files = os.listdir(data_dir)
        for fname in files:
            with open(data_dir + fname, 'rb') as f:
                batch_data = pickle.load(f)
                batch_X = np.zeros((len(batch_data), self.serp_len, self.input_size))
                batch_y = np.zeros((len(batch_data), self.serp_len, self.num_output))
                for (i, j) in enumerate(range(len(batch_data))):
                    matrix = batch_data[j].todense()
                    batch_X[i, :, 1:] = matrix[:, :-1]
                    batch_y[i, :] = matrix[:, -1]
                yield batch_X, batch_y
                del matrix, batch_X, batch_y, batch_data

    def load_lstm(self):
        """
        Function that loads LSTM model when weights are already saved
        """
        if os.path.isfile(self.weights_file):
            print "Found weights file, loading weights..."
            self.model.load_weights(self.weights_file)
            print "Weights loaded!", self.weights_file
        else:
            print "Error: did not find weights file!"

    def train_batch_lstm(self):
        """
        Function that trains LSTM model on batched pickles
        """
        print "Training the model..."
        train_starttime = time.time()

        # loop over epochs
        for epoch in range(self.epochs):
            #loop over steps
            batch_iterator = self.get_batch_pickle(self.train_dir)

            for step, (train_X, train_y) in enumerate(batch_iterator):
                #train on one batch
                perf_measures = self.model.train_on_batch(train_X, train_y)
                #print step
                if (step + 1) % self.print_step == 0 or step == 0:
                    print "Epoch: ", str(epoch + 1), "step", str(step+1), "train loss: ", perf_measures[0], 'time: ', time.time() - train_starttime
            print "Epoch: " + str(epoch + 1)
            print("Save temporary model")
            self.model.save_weights("TEMP_EPOCH_"+ str(epoch+1)+"_"+self.weights_file, True)


        print "Trained LSTM model in: %f seconds" % (time.time() - train_starttime)
        print('Saving the model...')
        self.model.save_weights(self.weights_file, True)

    def evaluate_batch_lstm(self):
        """
        Function that evaluates lstm on test batches
        """
        num_batches = 1500

        for directory in [self.train_dir, self.test_dir]:
            test_starttime = time.time()
            batch_iterator = self.get_batch_pickle(directory)
            bar = progressbar.ProgressBar(maxval=num_batches,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                               progressbar.Percentage()])
            perplexity_evaluator  = evaluate.Perplexity()
            cperplexity_evaluator = evaluate.ConditionalPerplexity()

            # loop over batches
            logsum = 0.
            perpsum = 0.
            #cperpsum = 0.
            #cppr_sum = np.zeros([10])
            ppr_sum = np.zeros([10])
            for step, (test_X, test_y) in bar(enumerate(batch_iterator)):
                if step == num_batches:
                    break
                logsum += self.model.evaluate(test_X, test_y, batch_size=64, verbose=0)[0]
                perp, perplexity_at_rank = perplexity_evaluator.evaluate(self.model.predict_proba(test_X, batch_size=64, verbose=0), test_y)
                #cperp, cperplexity_per_rank = cperplexity_evaluator.evaluate(test_X, test_y, self)
                perpsum += perp
                #cperpsum += cperp
                ppr_sum += perplexity_at_rank
                #cppr_sum += cperplexity_per_rank

            meanlog = logsum/float(num_batches)
            #meancperp = cperpsum/float(num_batches)
            #mean_cppr = cppr_sum/float(num_batches)
            meanperp = perpsum/float(num_batches)
            mean_ppr = ppr_sum/float(num_batches)
            print directory
            print 'perplexity_per_rank', mean_ppr
            #print 'conditional perplexity_per_rank', mean_cppr
            # print "Step: %d: Mean LogLikelihood: -%.3f, Perplexity: %.3f, Conditional Perplexity: %.3f. Duration: %.2f" % (
            #     step, meanlog, meanperp, meancperp, (time.time()-test_starttime))
            print "Step: %d: Mean LogLikelihood: -%.3f, Perplexity: %.3f. Duration: %.2f" % (
                step, meanlog, meanperp, (time.time()-test_starttime))

if __name__ == "__main__":
    lstmnn = LSTMNN()
    lstmnn.create_model()
    if lstmnn.train:
        lstmnn.train_batch_lstm()
    else:
        lstmnn.load_lstm()
        lstmnn.evaluate_batch_lstm()
