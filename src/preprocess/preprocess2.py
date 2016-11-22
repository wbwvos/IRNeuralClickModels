import numpy as np
import cPickle as pickle

import progressbar
import json
from scipy.sparse import coo_matrix


# 100000 queries should do the trick...


def get_index_from_click_pattern(click_pattern, location=1):
    index = (location - 1) * 1024
    index += int(''.join([str(i) for i in click_pattern]), 2)
    return index


def get_click_pattern_from_index(index):
    loc = index / 1024
    index -= (loc * 1024)
    click_pattern = map(int, np.binary_repr(index, width=10))
    return click_pattern, loc


if __name__ == "__main__":
    print "Opening dict ... "
    with open('train_100000_read.cpickle', 'rb') as f:
        query_list = pickle.load(f)
    bar = progressbar.ProgressBar(maxval=len(query_list),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    docs = {}
    queries = {}
    query_docs = {}

    for query in bar(query_list):
        query_doc = query_docs.get(query['query_id'], {})

        serps = query_doc.get('serps', [])
        serps.append({
            'doc_ids': query['doc_urls'],
            'click_pattern': query['click_pattern'],
        })

        query_doc['serps'] = serps

        # Append index to query representation
        l = queries.get(query['query_id'], [])
        l.append(get_index_from_click_pattern(query['click_pattern']))
        queries[query['query_id']] = l

        indices = [0] * 10
        for doc_location, doc_id in enumerate(query['doc_urls']):
            index = get_index_from_click_pattern(query['click_pattern'], doc_location + 1)
            indices[doc_location] = index

            # Append index to query-document representation
            doc_indices = query_doc.get(doc_id, [])
            doc_indices.append(index)
            query_doc[doc_id] = doc_indices

            # Append index to document representation
            l = docs.get(doc_id, [])
            l.append(index)
            docs[doc_id] = l

        query_docs[query['query_id']] = query_doc
    with open('query_docs_100000.json', 'w') as f2:
        json.dump(query_docs, f2)
    with open('query_100000.json', 'w') as f3:
        json.dump(queries, f3)
    with open('docs_100000.json', 'w') as f4:
        json.dump(docs, f4)
