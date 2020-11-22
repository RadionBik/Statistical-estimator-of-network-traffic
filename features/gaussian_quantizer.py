import logging
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from features.base_packet_transformer import BasePacketTransformer
from features.packet_scaler import PacketScaler
from settings import RANDOM_SEED

logger = logging.getLogger(__name__)


class GaussianQuantizer(BasePacketTransformer):
    def __init__(
            self,
            model_from: Optional[GaussianMixture] = None,
            model_to: Optional[GaussianMixture] = None,
            scaler=PacketScaler,
            pad_token_id=0,
            start_token_id=1,
    ):
        super().__init__(model_from, model_to, scaler)
        self._n_tokens_from = model_from.n_components if model_from else -1
        self._n_tokens_to = model_to.n_components if model_to else -1
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self._n_reserved_tokens = max([start_token_id, pad_token_id]) + 1

    def transform(self, features, client_direction_vector, prepend_with_init_tokens: int = 0):
        scaled_features = self.scaler.transform(features.copy())
        encoded = np.empty(len(scaled_features), dtype=np.int)
        from_idx = (client_direction_vector == True)
        to_idx = ~from_idx

        from_clusters = self.model_from.predict(scaled_features[from_idx])
        to_clusters = self.model_to.predict(scaled_features[to_idx])

        encoded[from_idx] = from_clusters + self._n_reserved_tokens
        encoded[to_idx] = to_clusters + (self._n_reserved_tokens + self._n_tokens_from)

        if prepend_with_init_tokens > 0:
            init_tokens = np.empty(prepend_with_init_tokens, dtype=np.int)
            init_tokens[:-1] = self.pad_token_id
            init_tokens[-1] = self.start_token_id
            encoded = np.concatenate([init_tokens, encoded])
        return encoded

    def inverse_transform(self, tokens, prob_sampling=False):
        eff_tokens = tokens[tokens >= self._n_reserved_tokens]
        from_idx = eff_tokens < (self._n_reserved_tokens + self._n_tokens_from)
        to_idx = eff_tokens >= (self._n_reserved_tokens + self._n_tokens_from)

        from_clusters = eff_tokens[from_idx] - self._n_reserved_tokens
        to_clusters = eff_tokens[to_idx] - self._n_reserved_tokens - self._n_tokens_from
        restored_features = np.empty((len(eff_tokens), self.model_to.n_features_in_))

        if prob_sampling:
            restored_features[from_idx] = multivariate_sampling_from_states(self.model_from, from_clusters)
            restored_features[to_idx] = multivariate_sampling_from_states(self.model_to, to_clusters)
        else:
            restored_features[from_idx] = self.model_from.means_[from_clusters]
            restored_features[to_idx] = self.model_to.means_[to_clusters]
        restored_features = self.scaler.inverse_transform(restored_features)
        return restored_features, from_idx

    def fit(self, features, client_direction_vector, **gmm_kwargs):
        return self._fit(GaussianMixture, features, client_direction_vector, **gmm_kwargs)

    @property
    def n_tokens(self):
        return self._n_reserved_tokens + self._n_tokens_from + self._n_tokens_to


def multivariate_sampling_from_states(model: GaussianMixture, states, random_state=RANDOM_SEED):
    rng = np.random.default_rng(seed=random_state)
    restored_features = np.zeros((len(states), model.n_features_in_))
    for i, state in enumerate(states.astype(np.int)):
        restored_features[i] = rng.normal(
            loc=model.means_[state],
            scale=np.sqrt(np.diag(model.covariances_[state])),
            size=model.n_features_in_
        )
    return restored_features
