import json
from scipy.sparse import csr_matrix


class SparseMatrixGenerator:
    def __init__(self, representation_set='1', fname='query_docs_queries_docs_100000.json'):

        # Open the json file
        with open(fname, 'r') as f:
            data = json.load(f)

        # Separate the document vectors and query vectors for set 2 and 3
        self.docs = data.pop('docs')
        self.queries = data.pop('queries')

        # Obtain the query dict and save occurences of indices
        self.query_dicts = data.pop('query_docs')

        # Set the representation set
        self.representation_set = representation_set

        # Delete unused variables
        del data
        # del query_dict

    def get_sparse_matrices(self, query_id='20369649'):
        """
        Returns a list of sparse matrices (one for each serp within the query id)
        :param query_id:
        :return list of sparse matrices:
        """
        sparse_matrices = []
        query_dict = self.query_dicts[query_id]
        serps = query_dict.pop('serps')
        doc_indices = query_dict

        q_indices = []
        if self.representation_set != '1':
            # add q vector to indices
            q_indices = [val + 10240 for val in self.queries[query_id]]

        for serp in serps:
            sparse_matrices.append(self.get_sparse_matrix_from_serp(serp, doc_indices, q_indices))
        return sparse_matrices

    def get_sparse_matrix_from_serp(self, serp, doc_indices, q_indices):
        data = []
        rows = []
        cols = []

        # Append q indices for set 2
        if self.representation_set != '1':
            q_index_counts = {index: q_indices.count(index) for index in q_indices}
            data.extend(q_index_counts.values() * 10)
            cols.extend(q_index_counts.keys() * 10)
            rows.extend([j for j in range(10) for i in range(len(q_index_counts))])

        # Append qd vector indices and d indices for set 3
        for row, doc_id in enumerate(serp['doc_ids']):
            if self.representation_set == '3':
                doc_indices.extend([val + 10240 + 1024 for val in self.docs[doc_id]])
            index_counts = {index: doc_indices[doc_id].count(index) for index in doc_indices[doc_id]}
            data.extend(index_counts.values())
            cols.extend(index_counts.keys())
            rows.extend([row] * len(index_counts.keys()))

        # Append interaction and label indices
        if self.representation_set == '1':
            interaction_index = 10240
        if self.representation_set == '2':
            interaction_index = 10240 + 1024
        if self.representation_set == '3':
            interaction_index = 10240 + 1024 + 10240

        interaction_row_indices = [i for i, j in enumerate(serp['click_pattern']) if j == 1]
        rows.extend(interaction_row_indices)
        data.extend([1] * len(interaction_row_indices))
        cols.extend([interaction_index] * len(interaction_row_indices))

        label_row_indices = [i for i, j in enumerate([0] + serp['click_pattern'][:-1]) if j == 1]
        rows.extend(label_row_indices)
        data.extend([1] * len(label_row_indices))
        cols.extend([interaction_index + 1] * len(label_row_indices))

        return csr_matrix((data, (rows, cols)), shape=(10, interaction_index + 2))

# sparseMatrixGenerator = SparseMatrixGenerator()
#
# with open('query_docs_queries_docs_100000.json', 'r') as f:
#     data = json.load(f)

# print sparseMatrixGenerator.get_sparse_matrices()

# start_time = time.time()
# for query in (data['queries'].keys()):
#     sparseMatrixGenerator.get_sparse_matrices(query_id=query)
#
# print "\nTime elapsed: " + str(time.time() - start_time)
