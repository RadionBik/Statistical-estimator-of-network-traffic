
import dataclasses
import json
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data.dataloader import DataLoader

import settings
import stat_metrics
from features.gaussian_quantizer import GaussianQuantizer
from pcap_parsing.parsed_fields import select_features
from tcn.model import TCNGenerator
from tcn_utils import StatesDataset, generate_states, get_model_size, get_eff_memory, get_init_seq

seed_everything(1)
TCN_DIR = settings.BASE_DIR / 'tcn'


@dataclasses.dataclass
class TrainConfig:
    es_patience: int = 5
    batch_size: int = 256
    optimizer: str = 'Adam'
    learning_rate: float = 0.001
    grad_clip: float = 1.0
    val_size: float = 0.2

    output_size: int = -1
    window_size: int = 128


@dataclasses.dataclass
class TCNConfig(TrainConfig):
    hidden_size: int = -1
    input_size: int = 1
    num_channels: list = None
    n_classes: int = -1
    n_levels: int = -1
    kernel_size: int = -1
    dropout: float = .0
    model_size: int = 0
    effective_memory: int = dataclasses.field(init=False)
    fft_loss: str = '1D'

    def __post_init__(self):
        self.effective_memory = get_eff_memory(filter_size=self.kernel_size, n_layers=self.n_levels)


def main():

    extr_stats = pd.read_csv(settings.BASE_DIR / 'traffic_dumps/iot_amazon_echo.pcap.csv', index_col=0)

    train_size = len(extr_stats) - len(extr_stats) // 3
    train_df, test_df = extr_stats.iloc[:train_size], extr_stats.iloc[train_size:]
    quantizer = GaussianQuantizer.from_pretrained(settings.BASE_DIR / 'obj/amazon_train')

    n_classes = quantizer.n_tokens
    n_levels = 7
    config = TCNConfig(
        hidden_size=n_classes,
        n_classes=n_classes,
        kernel_size=3,
        n_levels=n_levels,
        num_channels=[n_classes] * n_levels,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_states = quantizer.transform(*select_features(train_df), prepend_with_init_tokens=config.window_size)
    test_states = quantizer.transform(*select_features(test_df))

    train_size = len(train_states) - int(len(train_states) * config.val_size)

    train_states, val_states = train_states[:train_size], train_states[train_size:]

    train_dataset = StatesDataset(train_states, window=config.window_size, device=device)
    val_dataset = StatesDataset(val_states, window=config.window_size, device=device)

    pprint(config)

    model = TCNGenerator(config)

    config.model_size = get_model_size(model)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  drop_last=False,
                                  shuffle=False)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                drop_last=False,
                                shuffle=False)

    neptune_logger = NeptuneLogger(
        api_key=settings.NEPTUNE_API_TOKEN,
        offline_mode=True,
        close_after_fit=False,
        project_name='radion/TCN',
        experiment_name='revisited',
        params=dataclasses.asdict(config),
        upload_source_files=['model.py', 'tcn_utils.py']
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.005,
        patience=config.es_patience,
        verbose=False,
        mode='min'
    )

    trainer = Trainer(
        callbacks=[early_stop_callback],
        auto_lr_find=False,
        gpus=int(device == 'cuda'),
        gradient_clip_val=config.grad_clip,
        logger=neptune_logger,
        deterministic=True,
        check_val_every_n_epoch=1,
        # precision=16
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    # evaluation part
    tcn_states = generate_states(model,
                                 sample_number=len(test_states),
                                 window_size=config.window_size,
                                 device=device)

    res_metrics = stat_metrics.calc_stats(test_states, tcn_states)
    for k, v in res_metrics.items():
        neptune_logger.experiment.log_metric(k, v)

    neptune_logger.experiment.stop()

    with open(TCN_DIR / 'tcn_states.json', 'w') as jf:
        json.dump(tcn_states.tolist(), jf)


if __name__ == '__main__':
    main()
