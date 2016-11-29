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


class LogLikelihood(Evaluation):
    def __init__(self):
        pass

    def evaluate(self, prediction_probabilities, labels):
        loglikelihood = 0.0

        for i, probabilities_per_rank in enumerate(prediction_probabilities):
            label = [l[0] for l in labels[i]]
            probabilities_per_ra = [p[0] for p in probabilities_per_rank]
            ps = []
            for rank, click_prob in enumerate(probabilities_per_ra):
                if label[rank]:
                    p = click_prob
                else:
                    p = 1 - click_prob
                ps.append(p)
            log_click_probs = [math.log(prob) for prob in ps]
            loglikelihood += sum(log_click_probs)
        loglikelihood /= len(prediction_probabilities)

        return loglikelihood


class Perplexity(Evaluation):
    # markovi heeft clickmodels i.p.v pred en search_session i.p.v labels
    def __init__(self):
        pass

    def evaluate(self, prediction_probabilities, labels):
        # Initialize empty array

        perplexity_at_rank = [0.0] * 10  # Could give an error when labels increase

        for i, probability in enumerate(prediction_probabilities):
            label = labels[i]
            for rank, click_prob in enumerate(probability):
                if label[rank][0]:
                    p = click_prob[0]
                else:
                    p = 1 - click_prob[0]
                perplexity_at_rank[rank] += math.log(p, 2)

        perplexity_at_rank = [2 ** (-rank / len(prediction_probabilities)) for rank in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)

        return perplexity, perplexity_at_rank


if __name__ == "__main__":
    Evaluation = Perplexity()
    pred = [[[0.00001], [.00001]], [[.000001], [.00001]], [[.00001], [.0001]]]
    labels = [[[1], [1]], [[1], [1]], [[1], [1]]]
    #     pred = [[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]]
    #     labels = [[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]]
    p, _ = Evaluation.evaluate(pred, labels)
