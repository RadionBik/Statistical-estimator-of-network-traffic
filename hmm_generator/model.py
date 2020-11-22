from typing import Optional

from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

from features.packet_scaler import PacketScaler


class HMM(GaussianHMM, GaussianMixture):
    def _n_parameters(self) -> int:
        """ need to override for correct BIC estimation by adding transition matrix parameters """
        mixture_params = super()._n_parameters()
        return self.n_components ** 2 + mixture_params


class HMMGenerator:
    def __init__(
            self,
            model_from: Optional[HMM] = None,
            model_to: Optional[HMM] = None,
            scaler=PacketScaler
    ):
        self.model_from = model_from
        self.model_to = model_to
        self.scaler = scaler
