#!/usr/bin/env python
import cPickle as pickle
import glob
import os
import pprint

import progressbar
from scipy.sparse import csr_matrix
from utils import dict_batch_reader, list_batch_writer
import copy

__author__ = 'Wolf Vos, Casper Thuis, Alexander van Someren, Jeroen Rooijmans'


class SparseMatrixGenerator:
    def __init__(self, fname):

        data_dir = "../data/"
        if os.path.exists('/data'):
            data_dir = "/data/"
        self.query_dicts = {}
        self.filename = fname
        self.queries = dict_batch_reader(fname + "-q")
        self.docs = dict_batch_reader(fname + "-d")
        self.users = dict_batch_reader(data_dir + "user_indices_dict")

    def get_sparse_matrices(self, query_id='20369649', representation_set='1'):
        """
        Returns a list of sparse matrices (one for each serp within the query id)
        :param representation_set: Representation set as described in the paper
        :param query_id: The query ID as provided by the Yandex dataset
        :return list of sparse matrices:
        """
        sparse_matrices = []
        in_doc_indices = self.query_dicts[query_id]
        serps = in_doc_indices.pop('serps')
        user_id = in_doc_indices.pop('user_id')

        # To speed up computation
        if len(serps) > 200:
            serps = serps[0:200]

        in_rows = []
        in_data = []
        in_cols = []
        if representation_set != '1':
            # add q vector to indices
            q_indices = [val + 1024 for val in self.queries[query_id]]
            q_index_counts = {index: q_indices.count(index) for index in q_indices}
            in_data.extend(q_index_counts.values() * 10)
            in_cols.extend(q_index_counts.keys() * 10)
            in_rows.extend([j for j in range(10) for _ in range(len(q_index_counts))])

        if representation_set == '4':
            u_indices = [val + 1024 + 10240 + 1024 for val in self.users[user_id]]
            u_index_counts = {index: u_indices.count(index) for index in u_indices}
            in_data.extend(u_index_counts.values() * 10)
            in_cols.extend(u_index_counts.keys() * 10)
            in_rows.extend([j for j in range(10) for _ in range(len(u_index_counts))])

        for serp in serps:
            sparse_serp = self.get_sparse_matrix_from_serp(
                serp,
                in_doc_indices,
                representation_set,
                in_data, in_rows, in_cols)
            sparse_matrices.append(sparse_serp)
        return sparse_matrices

    def get_sparse_matrix_from_serp(self, serp, in_doc_indices, representation_set, in_data, in_rows, in_cols):

        rows = copy.copy(in_rows)
        cols = copy.copy(in_cols)
        data = copy.copy(in_data)
        doc_indices = copy.copy(in_doc_indices)

        # Append qd vector indices and d indices for set 3
        for row, doc_id in enumerate(serp['doc_ids']):  # (these are ranks)
            if representation_set == '3' or representation_set == '4':
                print doc_id in self.docs
                doc_indices[doc_id].extend([val + 10240 + 1024 for val in self.docs[doc_id]])
            index_counts = {index: doc_indices[doc_id].count(index) for index in doc_indices[doc_id]}
            data.extend(index_counts.values())  # Frequencies
            cols.extend(index_counts.keys())  # Location in qd (q) (d) vector
            rows.extend([row] * len(index_counts))  # Rank

        # Append interaction and label indices
        # if representation_set == '1':
        interaction_index = 10240
        if representation_set == '2':
            interaction_index = 10240 + 1024
        elif representation_set == '3':
            interaction_index = 10240 + 1024 + 10240
        elif representation_set == '4':
            interaction_index = 10240 + 1024 + 10240 + 1024

        interaction_row_indices = [i for i, j in enumerate([0] + serp['click_pattern'][:-1]) if j == 1]
        rows.extend(interaction_row_indices)
        data.extend([1] * len(interaction_row_indices))
        cols.extend([interaction_index] * len(interaction_row_indices))

        label_row_indices = [i for i, j in enumerate(serp['click_pattern']) if j == 1]
        rows.extend(label_row_indices)
        data.extend([1] * len(label_row_indices))
        cols.extend([interaction_index + 1] * len(label_row_indices))

        return csr_matrix((data, (rows, cols)), shape=(10, interaction_index + 2))

    def save_matrices_to_file(self, fname='sparse_matrices.cpickle', representation_set='1'):
        self.query_dicts = dict_batch_reader(fname + "-qd")
        sparse_matrices = []
        for query_id in self.queries.keys():
            sparse_matrices.extend(self.get_sparse_matrices(query_id, representation_set))
        with open(fname, 'w') as f:
            pickle.dump(sparse_matrices, f, -1)

    def save_matrices_to_files(self, fname='sparse_matrices.cpickle', representation_set='1'):

        sparse_matrices = []
        batch_size = 64
        filenames = glob.glob(self.filename + "-qd" + "/*.pickle")

        for i, filename in enumerate(filenames):
            print "Opening " + filename
            with open(filename, 'r') as f:
                self.query_dicts = pickle.load(f)
            print "Creating sparse matrices for " + str(len(self.query_dicts)) + " queries"

            bar = progressbar.ProgressBar(maxval=len(self.query_dicts),
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

            for query_id in bar(self.query_dicts):
                sparse_matrices.extend(self.get_sparse_matrices(query_id, representation_set))

            print "Size sparse matrices: " + str(len(sparse_matrices))
            print "Writing " + str(len(sparse_matrices) / batch_size - 1) + " batches of size " + str(
                batch_size) + "..."
            number_of_batches = len(sparse_matrices) / batch_size
            number_matrices_to_write = number_of_batches * batch_size
            if number_of_batches > 0:
                list_batch_writer(fname, sparse_matrices[0:number_matrices_to_write], batch_size=batch_size,
                                  extra_postfix=str(i))
                del sparse_matrices[0:number_matrices_to_write]
