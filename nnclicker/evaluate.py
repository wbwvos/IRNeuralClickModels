from abc import abstractmethod
import math

class Evaluation(object):
        """An abstract evaluation method for click models."""
        @abstractmethod
        def evaluate(self, click_model, search_sessions):
            """
            Evaluates the quality of the given click model using the given list of search sessions. This method must be implemented by subclasses.


            :param click_model: The click model to evaluate.
            :param search_sessions: The list of search sessions (also known as test set).
            :return: The quality of the click model, given the set of test search sessions.
            """
            pass

class Perplexity(Evaluation):


    # markovi heeft clickmodels i.p.v pred en search_session i.p.v labels
    def evaluate(self, serps, labels):
        # Initialize empty array

        perplexity_at_rank = [0.0] * 10 # Could give an error when labels increase

        for i,serp in enumerate(serps):
            # print "serp: " + str(serp)

            label = labels[i]
            for rank, click_prob in enumerate(serp):
                # print label[rank][0]
                # print "click_prob" + str(click_prob[0])
                if label[rank][0]:
                    p = click_prob[0]
                    # print "p" + str(p)
                else:
                    p = 1 - click_prob[0]
                    # print "p0"
                perplexity_at_rank[rank] += math.log(p,2)

        perplexity_at_rank = [2**(-rank/len(serp)) for rank in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank)/len(perplexity_at_rank)

        return perplexity, perplexity_at_rank


if __name__ == "__main__":
    Evaluation = Perplexity()
    pred = [[[0.00001],[.00001]],[[.000001],[.00001]],[[.00001],[.0001]]]
    labels = [[[1],[1]],[[1], [1]],[[1], [1]]]
#     pred = [[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]]
#     labels = [[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]]
    p, _ = Evaluation.evaluate(pred, labels)
