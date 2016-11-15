import numpy as np
import marshal as pickle
import json
import progressbar

import multiprocessing as multi
from multiprocessing import Manager

TOTAL_LINE_NUMBERS = 164439537


def read_session(first_line):
    filename = "../../ data / train"
    query_dict = {}
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
                        query_dict[query_id].append(
                            {
                                'click_pattern': click_pattern.tolist(),
                                'doc_ids': urls,
                                'terms': terms,
                            }
                        )
                    # print query_dict
                    glob_data.append(query_dict)


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
    print "Starting reading datafile..."
    p = multi.Pool(processes=8)
    p.map(read_session, indices)

    print "Datafile read. Starting processing dicts"
    for new_query_dict in glob_data:
        query_dict = update_query_dict(query_dict, new_query_dict)
    print "Writing json file"
    with open('query_dict.json', 'wb') as f:
        json.dump(query_dict, f)


if __name__ == "__main__":
    print "Opening indices ... "
    with open('meta_indices.pickle', 'rb') as f:
        indices = pickle.load(f)

    manager = Manager()
    glob_data = manager.list([])
    # print indices[0:len(indices)/10]
    read_data("../../data/train", indices=indices[0:len(indices) / 100])
    # print indices
    # read_data('../../data/train_10000.tab', limit=10)
    # find_indices('../../data/train', limit=10)
