#!/usr/bin/env python
import os.path
import cPickle as pickle

from utils import get_index_from_click_pattern
from utils import dict_batch_writer

__author__ = 'Wolf Vos, Casper Thuis, Alexander van Someren, Jeroen Rooijmans'


class NNclickParser(object):
    """
    A Parser for the Yandex database, available from the Yandex Personalized
    Web Search Challenge from Kaggle.
    (https://www.kaggle.com/c/yandex-personalized-web-search-challenge)
    """

    def __init__(self):
        self.TOTAL_NUMBER_OF_QUERIES = 65172853
        self.query_sessions = None
        self.query_docs = None
        self.queries = None
        self.docs = None

    def parse(self, session_filename, sessions_start=None, sessions_max=None):
        """
        Function that parses a file containing search sessions,
        formatted according to the Yandex Personalized Web Seach Database
        (https://www.kaggle.com/c/yandex-personalized-web-search-challenge/data)
        stores a list of search sessions.

        Session file contains either query of click actions:
        Format query action:
        SessionID; TimePassed; TypeOfRecord; SERPID; QueryID; ListOfTerms;
        ListOfURLsAndDomains

        Format search action:
        SessionID; TimePassed; TypeOfRecord; SERPID; URLID

        :params session_filename: name of file containing seach sessions
        :session_start: session_id from which the parser starts parsing
        :sessions_max: maximal number of search sessions that are parsed,
        if not set, all search sessions are parsed and returned
        """
        sessions_file = open(session_filename, "r")
        sessions = []
        session_id = None

        for line in sessions_file:
            if sessions_max and sessions_max <= len(sessions):
                break

            entry_array = line.strip().split("\t")

            # continue until session_start is reached
            if sessions_start > int(entry_array[0]):
                continue

            # check if line is query action
            if len(entry_array) >= 6 and entry_array[2] == "Q":
                click_pattern = 10 * [0]
                session_id = entry_array[0]
                query_id = entry_array[4]
                doc_urls = [comb.split(",")[0] for comb in entry_array[6::]]

                session = {"query_id": query_id, "doc_urls": doc_urls, "click_pattern": click_pattern}
                sessions.append(session)

            # if we have found a query, check if line is click action
            if session_id and len(entry_array) == 5 and entry_array[2] == "C":
                if entry_array[0] == session_id:
                    clicked_doc = entry_array[4]
                    if clicked_doc in doc_urls:
                        click_pattern[doc_urls.index(clicked_doc)] = 1
        # store sessions
        self.sessions = sessions

    def write_sessions(self, filename):
        """
        Function that writes list of search sessions to pickle file
        """
        with open(filename, "w") as f:
            pickle.dump(self.sessions, f, -1)

    def write_query_docs(self, filename):
        """
        Function that writes query doc dicts to pickle file
        """
        dict_batch_writer(self.query_docs, filename + "-qd")
        dict_batch_writer(self.queries, filename + "-q")
        dict_batch_writer(self.docs, filename + "-d")

    def load_sessions(self, filename):
        """
        Function that loads list of search sessions from pickle file
        """
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                query_sessions = pickle.load(f)
                self.query_sessions = query_sessions

    def load_query_docs(self, filename):
        """
        Function that loads dics with query documents from pickle file
        """
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                query_docs = pickle.load(f)
                self.query_docs = query_docs

    def create_data_dicts(self):
        """
        Function that creates dictionaries to store the preprocessed data
        """
        self.query_docs = {}
        self.queries = {}
        self.docs = {}

        for query in self.query_sessions:
            query_doc = self.query_docs.get(query["query_id"], {})
            serps = query_doc.get("serps", [])
            serps.append({
                "doc_ids": query["doc_urls"],
                "click_pattern": query["click_pattern"]
            })
            query_doc["serps"] = serps

            # append index to query representation (for set 2)
            query_indices = self.queries.get(query["query_id"], [])
            query_indices.append(get_index_from_click_pattern(query["click_pattern"]))
            self.queries[query["query_id"]] = query_indices

            for (doc_location, doc_id) in enumerate(query["doc_urls"]):
                index = get_index_from_click_pattern(query["click_pattern"], doc_location)

                # append index to query-document representation
                query_doc_indices = query_doc.get(doc_id, [])
                query_doc_indices.append(index)
                query_doc[doc_id] = query_doc_indices

                # append index to document representation (for set 3)
                doc_indices = self.docs.get(doc_id, [])
                doc_indices.append(index)
                self.docs[doc_id] = doc_indices

            self.query_docs[query["query_id"]] = query_doc
