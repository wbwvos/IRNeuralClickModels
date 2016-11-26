#!/usr/bin/env python
import os
import time
import logging
import sys

import preprocess
from sparse_matrix_creator import SparseMatrixGenerator

__author__ = 'Wolf Vos, Casper Thuis, Alexander van Someren, Jeroen Rooijmans'

# create logger
logging.basicConfig(filename='experiment.log', filemode='w',
                    format='%(asctime)s %(message)s',
                    stream=sys.stdout, level=logging.INFO)


def run_experiment():
    """
    Function that runs NNclickParser experiment, data is preprocessed if
    needed, resulting in the sparse matrixes that are fed to the lstm
    """

    # params
    sessions_start = 0
    sessions_max = 10000
    repr_set = "1"

    if len(sys.argv) >= 3:
        sessions_start = int(sys.argv[1])
        sessions_max = int(sys.argv[2])
    if len(sys.argv) == 4:
        repr_set = sys.argv[3]

    # files names
    data_dir = "../data/"
    datafile = "train"
    session_name = "train_%s-%s.pickle" % (sessions_start, sessions_max)
    query_name = "query_docs_"+session_name
    sparsematrix_name = "sparse_matrix_set"+repr_set+"_"+session_name

    # create parser
    logging.info("creating NNclickParser...")
    parser = preprocess.NNclickParser()

    # session preprocessing
    if os.path.isfile(data_dir+session_name):
        logging.info("found session file: %s" % (data_dir+session_name))
        if not parser.sessions:
            logging.info("loading session file...")
            parser.load_sessions(data_dir+session_name)
    else:
        logging.info("parsing session %d to %d from file: %s" % (
            sessions_start, sessions_max, data_dir+datafile))
        parser.parse(data_dir+datafile, sessions_start=sessions_start,
                     sessions_max=sessions_max)
        logging.info("writing session file: %s" % data_dir+session_name)
        parser.write_sessions(data_dir+session_name)

    # query docs preprocessing
    if os.path.isfile(data_dir+query_name):
        logging.info("found query docs file: %s" % (data_dir+query_name))
        if not parser.query_docs:
            logging.info("loading query docs file...")
            parser.load_query_docs(data_dir+query_name)
    else:
        if not parser.sessions:
            logging.info("loading session file...")
            parser.load_sessions(data_dir+session_name)
        logging.info("parsing query docs from file: %s" % (
            data_dir+session_name))
        parser.create_data_dicts()
        logging.info("deleting parser.sessions from memory...")
        # del parser.sessions
        logging.info("writing query docs file: %s" % (data_dir+query_name))
        parser.write_query_docs(data_dir+query_name)

    # sparse matrix preprocessing
    if os.path.isfile(data_dir+sparsematrix_name):
        logging.info("found sparse matrix file: %s" % (
            data_dir+sparsematrix_name))
    else:
        logging.info("creating SparseMatrixGenerator...")
        sparse_matrix_gen = SparseMatrixGenerator(fname=data_dir+query_name)
        logging.info("generating sparse matrixes...")
        sparse_matrix_gen.save_matrices_to_file(
            fname=data_dir+sparsematrix_name, representation_set=repr_set)

    # LSTM


if __name__ == "__main__":
    start_time = time.time()
    logging.info("starting experiment...")
    run_experiment()
    duration = time.time()-start_time
    logging.info("completed experiment! (duration: %f)" % duration)
