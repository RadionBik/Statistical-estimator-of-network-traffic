from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

from features.base_packet_transformer import BasePacketTransformer


class HMM(GaussianHMM, GaussianMixture):
    def _n_parameters(self) -> int:
        """ need to override for correct BIC estimation by adding transition matrix parameters """
        mixture_params = super()._n_parameters()
        return self.n_components ** 2 + mixture_params


class HMMGenerator(BasePacketTransformer):
    def sample_like(self):
        pass