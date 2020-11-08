import logging
import pathlib
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from features.packet_scaler import PacketScaler
from features.data_utils import save_obj, load_obj
from settings import RANDOM_SEED

logger = logging.getLogger(__name__)


class GaussianQuantizer:
    def __init__(
            self,
            gmm_from: Optional[GaussianMixture] = None,
            gmm_to: Optional[GaussianMixture] = None,
            pad_token_id=0,
            start_token_id=1,
            scaler=PacketScaler
    ):
        self.gmm_from: GaussianMixture = gmm_from
        self.gmm_to: GaussianMixture = gmm_to
        self._n_tokens_from = gmm_from.n_components if gmm_from else -1
        self._n_tokens_to = gmm_to.n_components if gmm_to else -1

        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.scaler = scaler()
        self._n_reserved_tokens = max([start_token_id, pad_token_id]) + 1

    @classmethod
    def from_pretrained(cls, load_path, **kwargs):
        load_path = pathlib.Path(load_path)
        gmm_to, _ = load_obj(load_path / 'gmm_to.pkl', by_stem=False)
        gmm_from, _ = load_obj(load_path / 'gmm_from.pkl', by_stem=False)
        return cls(gmm_from=gmm_from, gmm_to=gmm_to, **kwargs)

    def save_pretrained(self, save_dir):
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        save_obj(self.gmm_to, save_dir / 'gmm_to.pkl', by_stem=False)
        save_obj(self.gmm_from, save_dir / 'gmm_from.pkl', by_stem=False)

    def transform(self, features, client_direction_vector, prepend_with_init_tokens: int = 0):
        scaled_features = self.scaler.transform(features.copy())
        encoded = np.empty(len(scaled_features), dtype=np.int)
        from_idx = (client_direction_vector == True)
        to_idx = ~from_idx

        from_clusters = self.gmm_from.predict(scaled_features[from_idx])
        to_clusters = self.gmm_to.predict(scaled_features[to_idx])

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
        restored_features = np.empty((len(eff_tokens), self.gmm_to.n_features_in_))

        if prob_sampling:
            restored_features[from_idx] = multivariate_sampling_from_states(self.gmm_from, from_clusters)
            restored_features[to_idx] = multivariate_sampling_from_states(self.gmm_to, to_clusters)
        else:
            restored_features[from_idx] = self.gmm_from.means_[from_clusters]
            restored_features[to_idx] = self.gmm_to.means_[to_clusters]
        restored_features = self.scaler.inverse_transform(restored_features)
        return restored_features, from_idx

    def fit(self, features, client_direction_vector, **gmm_kwargs):
        scaled_features = self.scaler.transform(features.copy())

        from_out = get_gmm(scaled_features[client_direction_vector], **gmm_kwargs)
        to_out = get_gmm(scaled_features[~client_direction_vector], **gmm_kwargs)

        bic_dict = {}
        if gmm_kwargs.get('return_bic_dict'):
            self.gmm_from, bic_dict['from'] = from_out
            self.gmm_to, bic_dict['to'] = to_out
        else:
            self.gmm_from = from_out
            self.gmm_to = to_out

        self._n_tokens_from = self.gmm_from.n_components
        self._n_tokens_to = self.gmm_to.n_components
        if gmm_kwargs.get('return_bic_dict'):
            return self, bic_dict
        return self

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


def get_gmm(
        df,
        columns: tuple = None,
        n_comp=None,
        min_comp=5,
        max_comp=20,
        step_comp=1,
        sort_components=False,
        return_bic_dict=False,
        **kwargs
):

    if not n_comp:
        comp_range = range(min_comp, max_comp + 1, step_comp)
        logger.info(f'Started fitting GMM with auto number of components')
    else:
        comp_range = range(n_comp, n_comp + 1)
        logger.info(f'Started fitting GMM')

    comp_model = {}
    comp_bic = {}
    for comp in comp_range:
        if kwargs:
            model = BayesianGaussianMixture(n_components=comp,
                                            covariance_type="full",
                                            max_iter=500,
                                            tol=0.001,
                                            random_state=RANDOM_SEED,
                                            **kwargs)
        else:
            model = GaussianMixture(n_components=comp,
                                    covariance_type="full",
                                    random_state=RANDOM_SEED)

        df = df[columns] if columns else df

        model.fit(df)
        comp_model[comp] = model
        comp_bic[comp] = model.bic(df)
        logger.info(f'fit model with {comp} components, BIC={comp_bic[comp]}')

    best_comp, best_bic = min(comp_bic.items(), key=lambda x: x[1])
    best_model = comp_model[best_comp]
    logger.info(f'Best BIC={best_bic} is with {best_comp} components')

    if sort_components:
        sorted_indexes = np.linalg.norm(best_model.means_, axis=1).argsort()
        for attr in ['means_', 'covariances_', 'weights_']:
            setattr(best_model, attr, getattr(best_model, attr)[sorted_indexes])
        logger.info('reassigned components numbers according to their sorted norm')

    if return_bic_dict:
        return best_model, comp_bic
    return best_model
