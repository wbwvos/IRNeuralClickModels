import cPickle as pickle
from scipy.sparse import coo_matrix

with open('data_list.cpickle', 'rb') as f:
    data = pickle.load(f)

for d in data:
    matrix = d.todense()
