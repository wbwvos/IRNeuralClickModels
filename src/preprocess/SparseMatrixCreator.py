import json
import pprint

import time

import sys
from scipy.sparse import csr_matrix


class SparseMatrixGenerator:
    def __init__(self, fname='query_docs_queries_docs_100000.json'):

        # Open the json file
        with open(fname, 'r') as f:
            data = json.load(f)

        # Seperate the document vectors and query vectors for set 2 and 3
        self.docs = data.pop('docs')
        self.queries = data.pop('queries')

        # Obtain the query dict and save occurences of indices
        self.query_dicts = data.pop('query_docs')

        # Delete unused variables
        del data
        # del query_dict

    def get_sparse_matrices(self, query_id='20369649'):
        sparse_matrices = []
        query_dict = self.query_dicts[query_id]
        serps = query_dict.pop('serps')
        doc_indices = query_dict
        for serp in serps:
            sparse_matrices.append(self.get_sparse_matrix_from_serp(serp, doc_indices))
        return sparse_matrices

    def get_sparse_matrix_from_serp(self, serp, doc_indices):
        data = []
        rows = []
        cols = []
        for row, doc_id in enumerate(serp['doc_ids']):
            index_counts = {index: doc_indices[doc_id].count(index) for index in doc_indices[doc_id]}
            data.extend(index_counts.values())
            cols.extend(index_counts.keys())
            rows.extend([row] * len(index_counts.keys()))

        # TODO: append, set 2 & 3, append click_pattern as interacction
        return csr_matrix((data, (rows, cols)), shape=(10, 10241))  # TODO: right shapes


# with open("query_docs_100000.json", 'r') as f:
#     query_dict = json.load(f)

# pprint.pprint(query_dict['20369649'])

sparseMatrixGenerator = SparseMatrixGenerator()

start_time = time.time()
print sparseMatrixGenerator.get_sparse_matrices(query_id='20369649')
print "\nTime elapsed: " + str(time.time() - start_time)
