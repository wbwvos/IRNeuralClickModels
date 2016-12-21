from abc import abstractmethod
import math
import numpy as np
import itertools
import lstm
import progressbar


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
    """
    depricated, keras evalution is used for loglikelihood
    """
    def __init__(self):
        pass

    def evaluate(self, prediction_probabilities, labels):
        loglikelihood = 0.0
        eps = 1e-10

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
            log_click_probs = [math.log(prob+eps) for prob in ps]
            loglikelihood += sum(log_click_probs)  #TODO: CHECK OF WE DOOR 10 MOETEN DELEN!!!!
        loglikelihood /= len(prediction_probabilities)

        return loglikelihood

class Perplexity(Evaluation):
    def __init__(self):
        pass

    def evaluate(self, prediction_probabilities, labels):
        # epsilon
        eps = 1e-10
        # init perplexity per rank array
        perplexity_per_rank = 10*[0.]
        for i, pred in enumerate(prediction_probabilities):
            for j in range(len(perplexity_per_rank)):
                #print i,j, prediction_probabilities[i][j], labels[i][j]
                # if click
                if labels[i][j] == [0.]:
                    p = 1-(prediction_probabilities[i][j])
                else:
                    p = prediction_probabilities[i][j]
                perplexity_per_rank[j] += math.log(p, 2)
        perpl = 0
        #for perp in perplexity_per_rank:
        #    perpl += 2**(-perp/float(len(prediction_probabilities)))
        #perplexity = perpl/10.
        #perplexity_at_rank = np.array(perplexity_per_rank)/float(len(prediction_probabilities))
        perplexity_at_rank = [2 ** (-x / len(prediction_probabilities)) for x in perplexity_per_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_per_rank)

        return perplexity, perplexity_at_rank

class ConditionalPerplexity(Evaluation):
    def __init__(self):
        pass

    def evaluate(self, batch_X, batch_y, lstmnn):
        lst = np.array([np.array(list(reversed(l))) for l in list(
            itertools.product([[0.], [1.]], repeat=9))])
        lst = np.concatenate([np.zeros([512,1,1]), lst], axis=1)
        lst = np.reshape(lst, [512,10])

        TOTAL_RANKS = 10
        NUM_SERPS = batch_y.shape[0]
        perplexity_at_rank = [0.0] * TOTAL_RANKS

        # bar = progressbar.ProgressBar(maxval=NUM_SERPS,
        #                           widgets=[progressbar.Bar('=', '[', ']'), ' ',
        #                                    progressbar.Counter()])

        for s in range(NUM_SERPS):
            #print s
            data = batch_X[s]

            data = np.reshape(data, (1, 10, batch_X.shape[2]))
            labels = batch_y[s]

            data_m = np.tile(data.copy(), [512,1,1])
            data_m[:,:,-1] = lst
            probability = np.reshape(lstmnn.model.predict_proba(data_m, verbose=0).T, [10,512])
            probs = [[],[]]
            probs[0] = 1.0 - probability
            probs[1] = probability
            ppr = [0.] * TOTAL_RANKS
            for i in xrange(TOTAL_RANKS):
                if i == 0:
                    label = int(labels[i])
                    inter = np.zeros([2])
                    inter[0] = probs[0][0,i]
                    inter[1] = probs[1][0,i]
                    ppr[i] = inter[label]
                    prev_inter = inter.copy()
                else:
                    label = int(labels[i])
                    dims = [2] * (i + 1)
                    inter = np.zeros(dims)
                    c = 0.0
                    for idx in [list(reversed(l)) for l in list(itertools.product([0, 1], repeat=i+1))]:
                        inter[tuple(idx)] = probs[idx[0]][i, int(c)] * prev_inter[tuple(idx[1:])]
                        c += 0.5
                    ppr[i] = float(label) * np.sum(inter[1]) + float(1-label) * np.sum(inter[0])
                    prev_inter = inter.copy()
            for rank, click_prob in enumerate(ppr):
                perplexity_at_rank[rank] += math.log(click_prob, 2)

        perplexity_at_rank = [2 ** (-x / NUM_SERPS) for x in perplexity_at_rank]
        perplexity = sum(perplexity_at_rank) / len(perplexity_at_rank)
        return perplexity, perplexity_at_rank

if __name__ == "__main__":
    evaluator = ConditionalPerplexity()
    data_dir = "../data/sparse_matrix_set1_train_0-500000.pickle/"
    lstmnn_init = lstm.LSTMNN()
    lstmnn_init.create_model()
    batch_itr = lstmnn_init.get_batch_pickle(data_dir)
    for step, (batch_X, batch_y) in enumerate(batch_itr):
        evaluator.evaluate(batch_X, batch_y)
        break
