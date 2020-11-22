import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture

from features.base_packet_transformer import BasePacketTransformer
from pcap_parsing.parsed_fields import select_features, ParsedFields
from settings import RANDOM_SEED


class HMM(GaussianHMM, GaussianMixture):
    def _n_parameters(self) -> int:
        """ need to override for correct BIC estimation by adding transition matrix parameters """
        mixture_params = super()._n_parameters()
        return self.n_components ** 2 + mixture_params


class HMMGenerator(BasePacketTransformer):
    def fit(self, features, client_direction_vector, **kwargs):
        return self._fit(HMM, features, client_direction_vector, **kwargs)

    def sample_packets_like(self, reference_stats) -> pd.DataFrame:
        _, directions = select_features(reference_stats)
        from_idx = directions == True
        from_count, to_count = directions[from_idx].shape[0], directions[~from_idx].shape[0]
        packets_from, _ = self.model_from.sample(from_count, random_state=RANDOM_SEED)
        packets_to, _ = self.model_to.sample(to_count, random_state=RANDOM_SEED)
        # dumb concatenation, packet order is preserved direction-wise, not within the flow (!)
        packets_from = pd.DataFrame(self.scaler.inverse_transform(packets_from),
                                    columns=[ParsedFields.ps, ParsedFields.iat])
        packets_from[ParsedFields.is_source] = True

        packets_to = pd.DataFrame(self.scaler.inverse_transform(packets_to),
                                  columns=[ParsedFields.ps, ParsedFields.iat])
        packets_to[ParsedFields.is_source] = False
        return pd.concat([packets_from, packets_to], axis=0).reset_index(drop=True)
