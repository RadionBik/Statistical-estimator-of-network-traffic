import dataclasses
from typing import Type

from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

from .base_config import BaseConfig
from .datasets import ManyToManyDataset, ManyToOneDataset
from .rnn.config import RNNConfig
from .rnn.model import RNNGenerator
from .tcn.config import TCNConfig
from .tcn.model import TCNGenerator


@dataclasses.dataclass
class ClassMapper:
    dataset: Type[Dataset]
    model: Type[LightningModule]
    config: Type[BaseConfig]


SCENARIO_MAPPER = {
    'TCN': ClassMapper(
        ManyToManyDataset,
        TCNGenerator,
        TCNConfig
    ),
    'RNN': ClassMapper(
        ManyToOneDataset,
        RNNGenerator,
        RNNConfig
    )
}
