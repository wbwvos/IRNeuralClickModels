import sys
import time
import cPickle
from progressbar import ProgressBar
from progressbar import Bar
from progressbar import Percentage

TOTAL_NUMBER_OF_QUERIES = 65172853


def read_data(filename, query_limit=None):
    bar = ProgressBar(maxval=query_limit, widgets=[Bar("=", "[", "]"), " ",
                                                   Percentage()])
    sessions_file = open(filename, "r")
    sessions = []
    query_count = 0
    print "Started parsing file..."
    bar.start()
    for line in sessions_file:
        if query_count >= query_limit:
            break
        entry_array = line.strip().split("\t")

        # If the entry has 6 or more elements it is a query
        if len(entry_array) >= 6 and entry_array[2] == "Q":
            query_count += 1
            bar.update(query_count)
            session_id = entry_array[0]
            query_id = entry_array[4]
            doc_urls = [comb.split(",")[0] for comb in entry_array[6::]]
            click_pattern = [0]*10

            session = {}
            session["doc_urls"] = doc_urls
            session["query_id"] = query_id
            session["click_pattern"] = click_pattern
            sessions.append(session)

        # If the entry has 4 elements it is a click
        elif entry_array[2] == "C":
            if entry_array[0] == session_id:
                clicked_url = entry_array[4]
                if clicked_url in doc_urls:
                    index = doc_urls.index(clicked_url)
                    click_pattern[index] = 1
    bar.finish()
    print"Finished parsing!"
    return sessions


if __name__ == "__main__":
    # TODO: Add parsing file in sessions to create batches
    if len(sys.argv) != 2:
        print("Error: number of arguments")
        print("Usage: python read.py -query_limit -session_limit")
        sys.exit()
    query_limit = int(sys.argv[1])
    if query_limit > TOTAL_NUMBER_OF_QUERIES:
        print("Error: query limit is too high")
        print("Total amount of queries: %d" % TOTAL_NUMBER_OF_QUERIES)
        sys.exit()
    start_time = time.time()
    read_file = "../../data/train"
    dump_file = "read_train_%d.cpickle" % query_limit
    sessions = read_data(read_file, query_limit)
    print "Dumping pickle..."
    with open(dump_file, 'w') as f:
        cPickle.dump(sessions, f)
    print "Done dumping pickle!"
    end_time = time.time()
    print "Time elapsed: %s" % str(end_time-start_time)
