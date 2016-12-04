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
    batch_number = 0
    # for batch in chunks(dictionary, SIZE=number_of_batches):
    #     with open(os.path.join(fname, str(batch_number) + ".pickle"), 'w') as f:
    #         pickle.dump(batch, f)
    #     batch_number += 1

    for batch_number in bar(range(number_of_batches)):
        start = total_size / number_of_batches * batch_number
        end = total_size / number_of_batches * (batch_number + 1)
        with open(os.path.join(fname, str(batch_number) + ".pickle"), 'w') as f:
            pickle.dump({key: dictionary[key] for key in dictionary.keys()[start:end]}, f)
    print "Done."


def dict_batch_reader(fname):
    print "Batch reading " + fname + "..."
    filenames = glob.glob(fname + "/*.pickle")

    bar = progressbar.ProgressBar(maxval=len(filenames),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    dictionary = {}
    for filename in bar(filenames):
        with open(filename, 'r') as f:
            dictionary.update(pickle.load(f))
    return dictionary
