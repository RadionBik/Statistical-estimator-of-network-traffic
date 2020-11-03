import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def _normalize_by_rows(x: np.array):
    safe_x = x.copy()
    safe_x[safe_x == np.inf] = 10e6
    return normalize(safe_x, axis=1, norm='l1')


def _calc_transition_matrix(state_seq, state_numb):
    """ here the states are expected to be integers in [0, state_numb) """
    # init with values close-to-zero for smoothing
    transition_matrix = np.ones((state_numb, state_numb)) * 1e-6
    # count number of each possible transition
    for t in range(len(state_seq) - 1):
        j = state_seq[t]
        k = state_seq[t + 1]
        transition_matrix[j, k] += 1

    norm_trans_matrix = _normalize_by_rows(transition_matrix)
    logger.info(f'estimated transition matrix for {norm_trans_matrix.shape[0]} states')
    return norm_trans_matrix


def _calc_prior_probas(seq, state_numb):
    counts = np.zeros(state_numb)
    for state in range(state_numb):
        counts[state] = np.count_nonzero(seq == state)
    priors = counts / np.linalg.norm(counts, ord=1)
    logger.info('estimated vector of priors')
    return priors


class BaseGenerator:
    def fit(self, X):
        raise NotImplementedError

    def sample(self, n_sequences):
        raise NotImplementedError


class MarkovSequenceGenerator(BaseGenerator):
    def __init__(self):
        self.n_states = None
        self.transition_matrix = None
        self.init_priors = None
        self.index2value = {}
        self.value2index = {}
        self._states = None
        logger.info('init MarkovGenerator')

    def _map_values_to_indexes(self, X):
        self.value2index = {value: index for index, value in enumerate(np.unique(X))}
        self.index2value = {index: value for index, value in enumerate(np.unique(X))}
        X_mapped = np.array([self.value2index[val] for val in X])
        return X_mapped

    def _map_indexes_to_values(self, X_mapped):
        X = np.array([self.index2value[val] for val in X_mapped])
        return X

    def fit(self, X):
        self.n_states = len(np.unique(X))
        self._states = np.arange(self.n_states)

        X_mapped = self._map_values_to_indexes(X)

        self.transition_matrix = _calc_transition_matrix(X_mapped, self.n_states)
        self.init_priors = _calc_prior_probas(X_mapped, self.n_states)
        return self

    def sample(self, n_samples):
        assert n_samples > 0
        logger.info(f'started generating {n_samples} samples')
        sampled_seq = self._sample_sequence(n_samples)
        return self._map_indexes_to_values(sampled_seq)

    def _sample_sequence(self, n_samples):
        sampled = np.empty(n_samples, dtype=int)
        sampled[0] = np.random.choice(self._states, p=self.init_priors)
        for index in range(1, n_samples):
            sampled[index] = np.random.choice(self._states, p=self.transition_matrix[sampled[index-1], :])
        return sampled
