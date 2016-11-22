import cPickle as pickle
import time

import progressbar


# TOTAL_LINE_NUMBERS = 164439537


def read_data(filename, max_queries=100000, sessions_max=None):
    bar = progressbar.ProgressBar(maxval=max_queries,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    sessions_file = open(filename, "r")
    sessions = []
    query_count = 0

    bar.start()
    for line in sessions_file:
        if query_count >= max_queries:
            break

        entry_array = line.strip().split("\t")

        # If the entry has 6 or more elements it is a query
        if len(entry_array) >= 6 and entry_array[2] == "Q":
            query_count += 1
            bar.update(query_count)

            session_id = entry_array[0]
            query_id = entry_array[4]
            doc_urls = [comb.split(',')[0] for comb in entry_array[6::]]
            click_pattern = [0] * 10

            session = {}
            session['doc_urls'] = doc_urls
            session['query_id'] = query_id
            session['click_pattern'] = click_pattern
            sessions.append(session)

        # If the entry has 4 elements it is a click
        elif entry_array[2] == "C":
            if entry_array[0] == session_id:
                clicked_url = entry_array[4]
                if clicked_url in doc_urls:
                    index = doc_urls.index(clicked_url)
                    click_pattern[index] = 1

        # Else it is an unknown data format so leave it out
        else:
            continue

    return sessions


if __name__ == "__main__":
    start_time = time.time()
    sessions = read_data("../../data/train")
    with open('train_100000_read.cpickle', 'w') as f:
        pickle.dump(sessions, f)
    print "\nTime elapsed: " + str(time.time() - start_time)
