from abc import abstractmethod
import math
import numpy as np
import itertools


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
            probabilities_per_rank = [p[0] for p in probabilities_per_rank]
            ps = []
            for rank, click_prob in enumerate(probabilities_per_rank):
                if label[rank]:
                    p = click_prob
                else:
                    p = 1 - click_prob
                ps.append(p)
            log_click_probs = [math.log(prob) for prob in ps]
            loglikelihood += sum(log_click_probs)
        print "Length prediction_probabilities: " + str(len(prediction_probabilities))
        loglikelihood /= len(prediction_probabilities)

        return loglikelihood


class Perplexity(Evaluation):
    # markovi heeft clickmodels i.p.v pred en search_session i.p.v labels
    def __init__(self):
        pass

    def evaluate(self, prediction_probabilities, labels):
        # Initialize empty array
        perplexity_per_rank = [0.0] * 10  # Could give an error when labels increase

        for i, probabilities_per_rank in enumerate(prediction_probabilities):
            clicks = [item for sublist in labels[i] for item in sublist]
            probabilities = [item for sublist in probabilities_per_rank for item in sublist]
            for rank, probability in enumerate(probabilities):
                lst = [list(l) for l in list(itertools.product([0, 1], repeat=rank))]
                for previous_prob in prediction_probabilities[:rank]:
                    pass

        perplexity_per_rank = [2 ** (-rank_perplexity / len(prediction_probabilities)) for rank_perplexity in
                               perplexity_per_rank]
        perplexity = sum(perplexity_per_rank) / 10

        return perplexity, perplexity_per_rank


#
# class Evaluator:
#
#     def model_perplexity(self, predictions, clicks, is_training,batch_size,validation_size):
#         prediction = tf.add(tf.mul(predictions, clicks), tf.mul(tf.sub(1.0, clicks), tf.sub(1.0, predictions)))
#         norm = tf.cond(is_training, lambda:tf.div(-1.0, tf.to_float(batch_size)), lambda:tf.div(-1.0, tf.to_float(validation_size)))
#         sum_log = tf.reduce_sum(tf.div(tf.log(prediction), tf.log(2.0)), 0)
#         perplexity = tf.reduce_mean(tf.pow(2.0, tf.mul(norm, sum_log)))
#         tf.scalar_summary('perplexity', perplexity)
#         return perplexity


if __name__ == "__main__":
    Evaluation = Perplexity()
    pred = [[[0.6], [.01], [.000001], [.00001], [.00001], [.0001], [.0001], [.0001], [.0001], [.0001]],
            [[0.6], [.01], [.000001], [.00001], [.00001], [.0001], [.0001], [.0001], [.0001], [.0001]]]
    labels = [[[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]], [[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]]
    #     pred = [[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5],[0.5]]
    #     labels = [[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]]
    p, _ = Evaluation.evaluate(pred, labels)
