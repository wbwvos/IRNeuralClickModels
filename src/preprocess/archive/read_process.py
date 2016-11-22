import marshal as pickle
import progressbar
import numpy as np
from pymongo import MongoClient
import sys
import time

TOTAL_LINE_NUMBERS = 164439537


def get_index_from_click_pattern(click_pattern, location):
    index = (location - 1) * 1024
    index += int(''.join([str(i) for i in click_pattern]), 2)
    return index


def get_click_pattern_from_index(index):
    loc = index / 1024
    index -= index * 1024
    click_pattern = map(int, np.binary_repr(index, width=10))
    return click_pattern, loc


def process_query(query_id, serp, new_start=None):
    # print "Upserting query"
    query_dict_m.update({"query_id": query_id}, {'$push': serp}, True)
    if new_start is not None:
        processed_sessions.update({"_id": 0}, {'$addToSet': {"done": new_start}}, True)


def read_sessions(filename, new_start, new_end, number_to_be_processed):
    print number_to_be_processed
    bar = progressbar.ProgressBar(maxval=number_to_be_processed,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    j = 0
    process = False
    bar.start()
    with open(filename, "rb") as f:
        for k, l in enumerate(f):
            if k >= new_start:
                # print l
                if k == new_end:
                    print "At end"
                    print "Processed " + str(j) + " sessions."
                    break
                else:
                    l = l.split("\t")
                    # print l
                    if l[2] == "Q":
                        # Save old query + click pattern
                        if process:
                            if 'query_id' in locals():
                                process_query(
                                    query_id,
                                    {
                                        'click_pattern': click_pattern,
                                        'doc_ids': urls,
                                        'terms': terms,
                                    }
                                )

                            # Create new zero click pattern
                            query_id = l[4]
                            urls = [int(comb.split(',')[0]) for comb in l[6::]]
                            click_pattern = [0] * 10
                            terms = [int(i) for i in l[6].split(',')]
                    elif l[2] == "C":
                        if process:
                            urlid = int(l[4])

                            # Skip if clicked document is not in SERP of latest query
                            if urlid in urls:
                                position = urls.index(urlid)
                                click_pattern[position] = 1
                    elif l[1] == "M":
                        if process:
                            process_query(
                                query_id,
                                {
                                    'click_pattern': click_pattern,
                                    'doc_ids': urls,
                                    'terms': terms,
                                },
                                new_start
                            )
                        # if int(l[0]) in done:
                        #
                        if int(l[0]) in done:
                            process = False
                        else:
                            process = True
                        new_start = k
                        j += 1
                        # print j
                        bar.update(j + 1)


if __name__ == "__main__":
    NUMBER_OF_PROCESSES = 10
    print "Opening indices ... "
    with open('meta_indices.pickle', 'rb') as f:
        indices = pickle.load(f)
    process_number = int(sys.argv[1]) - 1

    total_indices = len(indices) / 10

    start = process_number * (total_indices / NUMBER_OF_PROCESSES)
    end = start + (total_indices / NUMBER_OF_PROCESSES)

    indices = set(indices[start:end])

    print "Initializing db connection..."

    client = MongoClient()
    db = client.ir2
    query_dict_m = db.query_dict
    processed_sessions = db.processed_sessions

    print "Checking for indices that were already processed..."

    processed_sessions.update({"_id": 0}, {'$addToSet': {"done": 99999999}}, True)
    done = set(processed_sessions.find_one()['done'])

    new_indices = indices.difference(done)

    number_to_be_processed = len(new_indices)

    print "Total assigned indices: " + str(len(indices))
    print "Already processed: " + str(len(done)) + " indices."
    print "Found " + str(len(indices) - len(new_indices)) + " duplicates."
    print "Continuing with " + str(len(new_indices)) + " indices."

    new_indices = list(new_indices)
    new_indices.sort()

    new_start = new_indices[0]
    new_end = new_indices[-1]

    print "First session at line: " + str(new_start)
    print "Last session at line: " + str(new_end)

    del new_indices
    del indices

    start_time = time.time()
    read_sessions("../../data/train", int(new_start), int(new_end), number_to_be_processed)
    print "Time elapsed: " + str(time.time() - start_time)
