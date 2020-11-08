import dataclasses


@dataclasses.dataclass
class BaseConfig:
    scenario: str
    n_classes: int

    es_patience: int = 3
    batch_size: int = 256
    optimizer: str = 'Adam'
    learning_rate: float = 0.001
    grad_clip: float = 1.0
    val_size: float = 0.2
    dropout: float = .0

    input_size: int = 1
    hidden_size: int = -1
    window_size: int = 128
    model_size: int = 0
    n_layers: int = 1

    def __post_init__(self):
        if self.hidden_size == -1:
            self.hidden_size = self.n_classes
