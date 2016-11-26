#!/usr/bin/env python
import os
import logging

import preprocess
from sparse_matrix_creator import SparseMatrixGenerator

__author__ = 'Wolf Vos, Casper Thuis, Alexander van Someren, Jeroen Rooijmans'

# create logger
logging.basicConfig(filename='experiment.log', filemode='w',
                    level=logging.INFO)


def run_experiment():
    """
    Function that runs NNclickParser experiment, data is preprocessed if
    needed, resulting in the sparse matrixes that are fed to the lstm
    """
    # files names
    data_dir = "../data/"
    datafile = "train"
    session_name = "train_0-100.pickle"
    query_name = "query_docs_0-100.pickle"

    # session params
    sessions_start = 0
    sessions_max = 100

    # set params
    repr_set = "1"

    # create parser
    logging.info("createing NNclickParser...")
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
    logging.info("Creating SparseMatrixGenerator...")
    sparse_matrix_gen = SparseMatrixGenerator(representation_set=repr_set,
                                              fname=data_dir+query_name)
    

if __name__ == "__main__":
    logging.info("starting experiment...")
    run_experiment()
