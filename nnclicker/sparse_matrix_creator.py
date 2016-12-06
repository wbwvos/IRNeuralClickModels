#!/usr/bin/env python
import cPickle as pickle
import glob

import progressbar
from scipy.sparse import csr_matrix
from utils import dict_batch_reader, list_batch_writer

__author__ = 'Wolf Vos, Casper Thuis, Alexander van Someren, Jeroen Rooijmans'


class SparseMatrixGenerator:
    def __init__(self, fname):
        self.query_dicts = {}
        self.queries = dict_batch_reader(fname + "-q")
        self.docs = dict_batch_reader(fname + "-d")

    def get_sparse_matrices(self, query_id='20369649', representation_set='1'):
        """
        Returns a list of sparse matrices (one for each serp within the query id)
        :param representation_set: Representation set as described in the paper
        :param query_id: The query ID as provided by the Yandex dataset
        :return list of sparse matrices:
        """
        sparse_matrices = []
        doc_indices = self.query_dicts[query_id]
        serps = doc_indices.pop('serps')

        q_indices = []
        data = []
        rows = []
        cols = []
        if representation_set != '1':
            # add q vector to indices
            q_indices = [val + 10240 for val in self.queries[query_id]]
            q_index_counts = {index: q_indices.count(index) for index in q_indices}
            data.extend(q_index_counts.values() * 10)
            cols.extend(q_index_counts.keys() * 10)
            rows.extend([j for j in range(10) for _ in range(len(q_index_counts))])

        for serp in serps:
            sparse_matrices.append(
                self.get_sparse_matrix_from_serp(
                    serp,
                    doc_indices,
                    representation_set,
                    data, rows, cols
                )
            )
        return sparse_matrices

    def get_sparse_matrix_from_serp(self, serp, doc_indices, representation_set, data, rows, cols):

        # Append qd vector indices and d indices for set 3
        for row, doc_id in enumerate(serp['doc_ids']):  # (these are ranks)
            if representation_set == '3':
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

        # filenames = glob.glob(fname + "-qd" + "/*.pickle")

        bar = progressbar.ProgressBar(maxval=len(self.queries.keys()),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        print "Creating matrices..."
        for query_id in bar(self.queries.keys()):
            sparse_matrices.extend(self.get_sparse_matrices(query_id, representation_set))
        print "Writing matrices..."
        list_batch_writer(fname + "_set_" + representation_set, sparse_matrices, batch_size=64)
