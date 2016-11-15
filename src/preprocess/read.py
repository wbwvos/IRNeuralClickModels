import itertools
import pandas as pd
import numpy as np
import marshal as pickle
import json
import progressbar
import threading

import sys

import time

TOTAL_LINE_NUMBERS = 164439537


def read_session(filename, first_line):
    # print "\n\nNEW SESSION!!!"
    query_dict = {}
    query_data = []
    click_pattern = []
    with open(filename, "rb") as f:
        for i, l in enumerate(f):
            if i >= first_line + 1:  # query
                l = l.split("\t")
                # print l
                if l[2] == "Q":
                    # Save old query + click pattern
                    if len(click_pattern) > 0:
                        if query_id not in query_dict.keys():
                            query_dict[query_id] = []
                        query_dict[query_id].append( \
                            {
                                'click_pattern': click_pattern.tolist(),
                                'doc_ids': urls,
                                'terms': terms,
                            })

                    # Create new zero click pattern
                    query_id = l[4]
                    urls = [int(comb.split(',')[0]) for comb in l[6::]]
                    click_pattern = np.zeros(10, dtype=np.int)
                    terms = [int(i) for i in l[6].split(',')]
                elif l[2] == "C":
                    urlid = int(l[4])

                    # Skip if clicked document is not in SERP of latest query
                    if urlid in urls:
                        position = urls.index(urlid)
                        click_pattern[position] = 1
                elif l[1] == "M":
                    if len(click_pattern) > 0:
                        if query_id not in query_dict.keys():
                            query_dict[query_id] = []
                        query_dict[query_id].append( \
                            {
                                'click_pattern': click_pattern.tolist(),
                                'doc_ids': urls,
                                'terms': terms,
                            })
                    # print query_dict
                    return query_dict




                    # current_query_doc_indices = [str(query_id) + '-' + str(urlid) for urlid in urls]
                    # for current_query_doc_index in current_query_doc_indices:
                    #     if not current_query_doc_index in query_doc_data.index:
                    #         query_doc_data.loc[current_query_doc_index] = np.zeros(10240)
                    # print current_query_doc_indices
                    # print "------------"
                    #
                    # elif l[2] == "C":
                    #     # SessionID TimePassed TypeOfRecord SERPID URLID
                    #     urlid = int(l[4])
                    #     position = urls.index(urlid)
                    #     print position
                    #     query_doc_id = query_id + '-' + str(urlid)
                    #
                    #     pass
                    # elif l[1] == "M":
                    #     # SessionID TypeOfRecord Day USERID
                    #     meta.append(l[2:4])
                    #     current_session_id = l[0]
                    # if i == limit:
                    #     break
    pass


def update_query_dict(query_dict, new_query_dict):
    for key in new_query_dict:
        if key in query_dict.keys():
            new_query_dict[key].extend(query_dict[key])
    query_dict.update(new_query_dict)
    return query_dict


def read_data(filename, indices):
    print "Found " + str(len(indices)) + " sessions."
    bar = progressbar.ProgressBar(maxval=len(indices),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    query_dict = {}
    my_lock = threading.Lock()
    # meta = []
    # cols = [str(list(i)) + '-' + str(j) for j in range(10) for i in list(itertools.product([0, 1], repeat=10))]
    # query_doc_data = pd.DataFrame(columns=[cols])
    print "Starting reading datafile..."
    for index in bar(indices):
        new_query_dict = read_session(filename, index)
        if new_query_dict is not None:
            query_dict = update_query_dict(query_dict, new_query_dict)
    print "Writing json file"
    with open('query_dict.json', 'wb') as f:
        json.dump(query_dict, f)
        # print "Size: " + str(sys.getsizeof(query_dict))
        # meta = pd.DataFrame(meta)
        # meta.columns = ["Day", "USERID"]


def find_indices(filename, limit=164439537):
    indices = []
    with open(filename, "rb") as f:
        for i, l in enumerate(f):
            l = l.split("\t")
            if l[1] == "M":
                indices.append(int(i))
    with open('meta_indices.pickle', 'wb') as f:
        pickle.dump(indices, f)


if __name__ == "__main__":
    print "Opening indices ... "
    with open('meta_indices.pickle', 'rb') as f:
        indices = pickle.load(f)
    # print indices[0:len(indices)/10]
    read_data("../../data/train", indices=indices[0:len(indices) / 100])
    # print indices
    # read_data('../../data/train_10000.tab', limit=10)
    # find_indices('../../data/train', limit=10)
