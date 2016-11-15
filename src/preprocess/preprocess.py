import json
import numpy as np
import cPickle as pickle

import progressbar
from scipy.sparse import coo_matrix


def get_index_from_click_pattern(click_pattern, location):
    index = (location - 1) * 1024
    index += int(''.join([str(i) for i in click_pattern]), 2)
    return index


def get_click_pattern_from_index(index):
    loc = index / 1024
    index -= index * 1024
    click_pattern = map(int, np.binary_repr(index, width=10))
    return click_pattern, loc


if __name__ == "__main__":
    print "Opening dict ... "
    with open('query_dict_10000.json', 'rb') as f:
        query_dict = json.load(f)
    dataset = {}
    datalist = []
    print "Processing " + str(len(query_dict.keys())) + " queries."
    bar = progressbar.ProgressBar(maxval=len(query_dict.keys()),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    for query, serps in bar(query_dict.iteritems()):
        query_data = []
        for serp in serps:
            query_serp_data = np.zeros([10, 10242], dtype=int)
            query_serp_data[:, -2] = [0] + serp['click_pattern'][:-1]
            query_serp_data[:, -1] = serp['click_pattern']

            for doc_location, doc_id in enumerate(serp['doc_ids']):
                for other_serp in serps:
                    if doc_id in other_serp['doc_ids']:
                        location = other_serp['doc_ids'].index(doc_id) + 1
                        query_serp_data[
                            doc_location,
                            get_index_from_click_pattern(other_serp['click_pattern'], location)
                        ] += 1
            serp_data = coo_matrix(query_serp_data)
            # np.savetxt("test2.csv", query_serp_data, delimiter=",", precision='i')
            datalist.append(serp_data)
            query_data.append(serp_data)
        dataset[query] = query_data
    print "Saving pickle files..."
    with open('data_list.cpickle', 'wb') as f:
        pickle.dump(datalist, f)
    with open('data_per_query.cpickle', 'wb') as f:
        pickle.dump(dataset, f)
