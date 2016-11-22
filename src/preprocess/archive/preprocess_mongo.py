import json
import numpy as np
import cPickle as pickle

import progressbar
from pymongo import MongoClient
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

    client = MongoClient()
    db = client.ir2_from_server
    query_dict_m = db.query_dict

    cursor = query_dict_m.find({})

    query_count = cursor.count()

    datalist = []

    print "Processing " + str(query_count) + " queries."
    bar = progressbar.ProgressBar(maxval=query_count,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    for index in bar(range(query_count)):
        query_data = []
        doc = cursor[index]
        # print len(doc['terms'])
        for serp_index in range(len(doc['terms'])):
            query_serp_data = np.zeros([10, 10242], dtype=int)
            query_serp_data[:, -2] = [0] + doc['click_pattern'][serp_index][:-1]
            query_serp_data[:, -1] = doc['click_pattern'][serp_index]

            # print doc['doc_ids']
            for doc_location, doc_id in enumerate(doc['doc_ids'][serp_index]):
                for other_serp_index in range(len(doc['terms'])):
                    if doc_id in doc['doc_ids'][other_serp_index]:
                        location = doc['doc_ids'][other_serp_index].index(doc_id) + 1
                        query_serp_data[
                            doc_location,
                            get_index_from_click_pattern(doc['click_pattern'][other_serp_index], location)
                        ] += 1
            serp_data = coo_matrix(query_serp_data)
            # np.savetxt("test2.csv", query_serp_data, delimiter=",", precision='i')
            datalist.append(serp_data)
            # query_data.append(serp_data)
            # dataset[query] = query_data
    print "Saving pickle files..."
    with open('data_list_from_server.cpickle', 'wb') as f:
        pickle.dump(datalist, f)
        # with open('data_per_query_from_server.cpickle', 'wb') as f:
        #     pickle.dump(dataset, f)
