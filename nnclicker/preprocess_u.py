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
        self.user_dict = {}
        self.user_indices_dict = {}

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
        session_id = None
        user_dict = {}

        sessions = 0
        for line in sessions_file:
            if sessions_max and (sessions_max - sessions_start) <= sessions:
                break

            entry_array = line.strip().split("\t")

            # continue until session_start is reached
            if sessions_start > int(entry_array[0]):
                continue

            if len(entry_array) <= 5 and entry_array[1] == "M":
                user_id = entry_array[3]
                if session_id is not None:
                    user_dict[user_id] = current_user_list
                current_user_list = []

            # check if line is query action
            if len(entry_array) >= 6 and entry_array[2] == "Q":
                click_pattern = 10 * [0]
                session_id = entry_array[0]
                doc_urls = [comb.split(",")[0] for comb in entry_array[6::]]
                sessions += 1
                current_user_list.append(click_pattern)

            # if we have found a query, check if line is click action
            if session_id and len(entry_array) == 5 and entry_array[2] == "C":
                if entry_array[0] == session_id:
                    clicked_doc = entry_array[4]
                    if clicked_doc in doc_urls:
                        click_pattern[doc_urls.index(clicked_doc)] = 1
        self.user_dict = user_dict

    def write_sessions(self, filename):
        """
        Function that writes list of search sessions to pickle file
        """
        print "Writing " + filename
        with open(filename, "w") as f:
            pickle.dump(self.user_dict, f, -1)

    def write_user_indices_dict(self, filename):
        """
        Function that writes query doc dicts to pickle file
        """
        print "Number of users: " + str(len(self.user_indices_dict))
        dict_batch_writer(self.user_indices_dict, filename + "-u")

    # def load_sessions(self, filename):
    #     """
    #     Function that loads list of search sessions from pickle file
    #     """
    #     if os.path.isfile(filename):
    #         with open(filename, "rb") as f:
    #             query_sessions = pickle.load(f)
    #             self.query_sessions = query_sessions

    def load_user_dict(self, filename):
        """
        Function that loads dics with query documents from pickle file
        """
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                user_dict = pickle.load(f)
                self.user_dict = user_dict

    def create_user_indices_dict(self):
        """
        Function that creates dictionaries to store the preprocessed data
        """
        self.user_indices_dict = {}

        print "len self.user_dict: ", self.user_dict
        for user_id in self.user_dict:
            user_indices = self.user_indices_dict.get(user_id, [])

            for click_pattern in self.user_dict[user_id]:
                user_indices.append(get_index_from_click_pattern(click_pattern))

            self.user_indices_dict[user_id] = user_indices
