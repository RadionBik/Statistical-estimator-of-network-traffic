import dataclasses

from .tcn_utils import get_eff_memory
from ..base_config import BaseConfig


@dataclasses.dataclass
class TCNConfig(BaseConfig):
    kernel_size: int = 3

    def __post_init__(self):
        super().__post_init__()
        self.num_channels: list = [self.n_classes] * self.n_layers
        self.effective_memory = get_eff_memory(filter_size=self.kernel_size, n_layers=self.n_layers)
