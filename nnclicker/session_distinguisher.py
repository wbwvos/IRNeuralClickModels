class SessionDistinguisher:
    def __init__(self):
        pass

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
                session = {}
                session["query_id"] = query_id
                session["doc_urls"] = doc_urls
                session["click_pattern"] = click_pattern
                sessions.append(session)

            # if we have found a query, check if line is click action
            if session_id and len(entry_array) == 5 and entry_array[2] == "C":
                if entry_array[0] == session_id:
                    clicked_doc = entry_array[4]
                    if clicked_doc in doc_urls:
                        index = doc_urls.index(clicked_doc)
                        click_pattern[index] = 1
        # store sessions
        self.sessions = sessions
