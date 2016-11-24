import json
import pprint
from scipy.sparse import csr_matrix


class SparseMatrixGenerator:
    def __init__(self, fname='query_docs_queries_docs_100000.json'):
        with open(fname, 'r') as f:
            data = json.load(f)
        query_dict = data.pop('query_dict')
        self.serps = qeuery_dict.pop('serps')
        self.doc_indices = {}
        for doc_id, indices in qeuery_dict.iteritems():
            self.doc_indices[doc_id] = {index: indices.count(index) for index in indices}

    def get_sparse_matrices(self):
        sparse_matrices = []
        for serp in self.serps:
            sparse_matrices.append(self.get_sparse_matrix_from_serp(serp))
        return sparse_matrices

    def get_sparse_matrix_from_serp(self, serp):
        data = []
        rows = []
        cols = []
        for row, doc_id in enumerate(serp['doc_ids']):
            index_counts = self.doc_indices[doc_id]
            data.extend(index_counts.values())
            cols.extend(index_counts.keys())
            rows.extend([row] * len(index_counts.keys()))
        return csr_matrix((data, (rows, cols)), shape=(10, 10241))  # TODO: right shapes
        # TODO: append, set 2 & 3, append click_pattern as interacction


with open("query_docs_100000.json", 'r') as f:
    query_dict = json.load(f)

pprint.pprint(query_dict['20369649'])

sparseMatrixGenerator = SparseMatrixGenerator(query_dict['20369649'])
print sparseMatrixGenerator.get_sparse_matrices()
