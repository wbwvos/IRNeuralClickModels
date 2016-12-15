#!/usr/bin/env python
import errno
import glob

import numpy as np
import os
import cPickle as pickle
import progressbar
from itertools import islice

__author__ = 'Wolf Vos, Casper Thuis, Alexander van Someren, Jeroen Rooijmans'


def get_index_from_click_pattern(click_pattern, location=1):
    """
    Function that returns index given a click pattern
    """
    index = (location) * 1024
    index += int(''.join([str(i) for i in click_pattern]), 2)
    return index


def get_click_pattern_from_index(index):
    """
    Function that returns click pattern from given index
    """
    loc = index / 1024
    index -= (loc * 1024)
    click_pattern = map(int, np.binary_repr(index, width=10))
    return click_pattern, loc


# def chunks(data, SIZE=10):
#     it = iter(data)
#     for i in xrange(0, len(data), SIZE):
#         yield {k: data[k] for k in islice(it, SIZE)}


def dict_batch_writer(dictionary, fname, number_of_batches=10):
    print "Batch writing " + fname + "..."
    bar = progressbar.ProgressBar(maxval=len(dictionary),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    if not os.path.exists(fname):
        try:
            os.makedirs(fname)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    total_size = len(dictionary.keys())
    for batch_number in bar(range(number_of_batches)):
        start = total_size / number_of_batches * batch_number
        end = total_size / number_of_batches * (batch_number + 1)
        with open(os.path.join(fname, str(batch_number) + ".pickle"), 'w') as f:
            pickle.dump({key: dictionary[key] for key in dictionary.keys()[start:end]}, f, -1)
    print "Done. Succesfully wrote " + str(batch_number + 1) + " batches."


def dict_batch_reader(fname):
    print "Batch reading " + fname + "..."
    filenames = glob.glob(fname + "/*.pickle")

    # bar = progressbar.ProgressBar(maxval=len(filenames),
    #                               widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    dictionary = {}
    print filenames
    for filename in filenames:
        with open(filename, 'r') as f:
            dictionary.update(pickle.load(f))
    return dictionary


def list_batch_writer(fname, output_list, batch_size=64, extra_postfix=''):
    print "Batch writing " + fname + "..."

    if not os.path.exists(fname):
        try:
            os.makedirs(fname)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    total_size = len(output_list)
    print "Total size: " + str(total_size)

    bar = progressbar.ProgressBar(maxval=total_size / batch_size,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    for batch_number in bar(range(total_size / batch_size)):
        start = batch_number * batch_size
        end = (batch_number + 1) * batch_size

        if end > total_size:
            end = total_size

        with open(os.path.join(fname, extra_postfix + "-" + str(batch_number) + ".pickle"), 'w') as f:
            pickle.dump(output_list[start:end], f, -1)
    print "Done. Succesfully wrote " + str(batch_number) + " batches of size " + str(batch_size) + "."
