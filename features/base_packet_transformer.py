import pathlib
from typing import Optional

from features.auto_select_model_by_bic import auto_select_model_by_bic
from features.data_utils import load_obj, save_obj
from features.packet_scaler import PacketScaler


class BasePacketTransformer:
    def __init__(
            self,
            model_from: Optional = None,
            model_to: Optional = None,
            scaler=PacketScaler
    ):

        self.model_from = model_from
        self.model_to = model_to
        self.scaler = scaler()

    @classmethod
    def from_pretrained(cls, load_path, **kwargs):
        load_path = pathlib.Path(load_path)
        gmm_to, _ = load_obj(load_path / 'to.pkl', by_stem=False)
        gmm_from, _ = load_obj(load_path / 'from.pkl', by_stem=False)
        return cls(model_from=gmm_from, model_to=gmm_to, **kwargs)

    def save_pretrained(self, save_dir):
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        save_obj(self.model_to, save_dir / 'to.pkl', by_stem=False)
        save_obj(self.model_from, save_dir / 'from.pkl', by_stem=False)

    def _fit(self, model_class, features, client_direction_vector, **model_kwargs):
        scaled_features = self.scaler.transform(features.copy())

        from_out = auto_select_model_by_bic(model_class, scaled_features[client_direction_vector], **model_kwargs)
        to_out = auto_select_model_by_bic(model_class, scaled_features[~client_direction_vector], **model_kwargs)

        bic_dict = {}
        self.model_from = from_out[0]
        self.model_to = to_out[0]

        self._n_tokens_from = self.model_from.n_components
        self._n_tokens_to = self.model_to.n_components

        if model_kwargs.get('return_bic_dict'):
            bic_dict['from'] = from_out[1]
            bic_dict['to'] = to_out[1]
            return self, bic_dict

        return self
