# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import logging
import sys
from pprint import pprint

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s %(name)s:%(lineno)d %(levelname)s - %(message)s')

# %%
sys.path.append("..")
import mixture_models
import plotting
import settings
import pcap_parser
import utils
import stat_metrics

# sys.path.pop()

# %% md [markdown]
# ## Initialize parameters

# %%
import dataclasses


@dataclasses.dataclass
class ExperimentConfig:
    # scenario: str = 'amazon'
    # pcap_file: str = settings.BASE_DIR / 'traffic_dumps/iot_amazon_echo.pcap'
    # pcap_traffic_kind: utils.TrafficObjects = utils.TrafficObjects.MAC
    # device_identifier: str = '44:65:0d:56:cc:d3'
    scenario: str = 'skype'
    pcap_file: str = settings.BASE_DIR / 'traffic_dumps/skypeLANhome.pcap'
    pcap_traffic_kind: utils.TrafficObjects = utils.TrafficObjects.FLOW
    window_size: int = 295
    traffic_direction: str = ''
    hidden_size: int = -1
    n_classes: int = -1
    n_levels: int = 5
    kernel_size: int = 10
    epochs: int = 300
    dropout: float = 0.0
    batch_size: int = 64
    optimizer: str = 'Adam'
    learning_rate: float = 0.0005
    grad_clip: float = 1.0
    train_val_splits: int = 10
    log_interval: int = 50
    sort_components: bool = True
    model_size: int = 0


CONFIG = ExperimentConfig()

# %%
traffic_dfs = pcap_parser.get_traffic_features(CONFIG.pcap_file,
                                               type_of_identifier=CONFIG.pcap_traffic_kind,
                                               # identifiers=[CONFIG.device_identifier],
                                               percentiles=(1, 99),
                                               min_samples_to_estimate=100)[0]

norm_traffic, scalers = utils.normalize_dfs(traffic_dfs, std_scaler=False)

# %%
useTrainedGMM = 1
gmm_model_path = f'{CONFIG.scenario}_gmm'
if not useTrainedGMM:
    gmm_models = mixture_models.get_gmm(norm_traffic, sort_components=CONFIG.sort_components)
    utils.save_obj(gmm_models, gmm_model_path)
else:
    gmm_models = utils.load_obj(gmm_model_path)

gmm_states = mixture_models.get_mixture_state_predictions(gmm_models, norm_traffic)

# %%
for device, direction, gmm_model in utils.iterate_2layer_dict(gmm_models):
    states = gmm_states[device][direction]
    CONFIG.traffic_direction = direction
    # save states to json
    json_states = f'gmm_{CONFIG.scenario}_{CONFIG.traffic_direction}.json'
    with open(json_states, 'w') as jsf:
        json.dump(states.tolist(), jsf)

plotting.plot_gmm_components(gmm_model)

# %%

# %%

# %% md [markdown]
# ## Load saved GMM states

# %%
json_states = (settings.BASE_DIR / 'tcn' / 'gmm_skype_from.json').as_posix()
gmm_model_path = (settings.BASE_DIR / 'obj' / 'skype_gmm.pkl').as_posix()
with open(json_states, 'r') as jsf:
    states = np.array(json.load(jsf))

# %% md [markdown]
# ## Train model

# %%
import scipy
import copy

import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from base_model import TemporalConvNet

from tcn_utils import StatesDataset, generate_states, get_model_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = StatesDataset(states, window=CONFIG.window_size, device=device)

CONFIG.hidden_size = dataset.n_states
CONFIG.n_classes = dataset.n_states


class TCN(pl.LightningModule):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = torch.nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1.transpose(1, 2))

    def configure_optimizers(self):
        return getattr(torch.optim, CONFIG.optimizer)(self.parameters(), lr=CONFIG.learning_rate)

    def _calc_loss(self, output, target, n_classes):
        return F.cross_entropy(output.view(-1, n_classes), target.view(-1))

    def _calc_accuracy(self, output, target, n_classes):
        pred = output.view(-1, n_classes).data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).sum()
        counter = output.view(-1, n_classes).size(0)
        return 100. * correct / counter

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.unsqueeze(1))
        loss = self._calc_loss(y_hat, y, CONFIG.n_classes)
        accuracy = self._calc_accuracy(y_hat, y, CONFIG.n_classes)

        return {'loss': loss, 'accuracy': accuracy}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        logs = {'log': {'train_loss': avg_loss, 'train_accuracy': avg_accuracy}}
        return logs

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.unsqueeze(1).contiguous())
        loss = self._calc_loss(y_hat, y, CONFIG.n_classes)
        accuracy = self._calc_accuracy(y_hat, y, CONFIG.n_classes)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        generated_states = generate_states(self, dataset,
                                           sample_number=1000,
                                           shuffle=False,
                                           prepend_init_states=False,
                                           device=device
                                           )
        try:
            kl_distance = stat_metrics.get_KL_divergence_pdf(states, generated_states)
        except Exception as e:
            logger.error(e)
            kl_distance = np.nan
        ws_distance = scipy.stats.wasserstein_distance(states, generated_states)

        logs = {'val_loss': avg_loss,
                'val_accuracy': avg_accuracy,
                'kl_dist': kl_distance,
                'wasserstein_dist': ws_distance
                }
        neptune_logs = copy.deepcopy(logs)
        logs.update({'log': neptune_logs})
        return logs


model = TCN(1, output_size=CONFIG.n_classes,
            num_channels=[CONFIG.hidden_size] * CONFIG.n_levels,
            kernel_size=CONFIG.kernel_size,
            dropout=CONFIG.dropout)

CONFIG.model_size = get_model_size(model)
pprint(dataclasses.asdict(CONFIG))

val_num = len(dataset) // CONFIG.train_val_splits

train_dataset, val_dataset = random_split(dataset, lengths=(len(dataset) - val_num, val_num))

train_dataloader = DataLoader(train_dataset,
                              batch_size=CONFIG.batch_size,
                              drop_last=True,
                              shuffle=False)

val_dataloader = DataLoader(val_dataset,
                            batch_size=CONFIG.batch_size,
                            drop_last=True,
                            shuffle=False)


neptune_logger = NeptuneLogger(api_key=settings.NEPTUNE_API_TOKEN,
                               offline_mode=False,
                               close_after_fit=False,
                               project_name='radion/TCN',
                               params=dataclasses.asdict(CONFIG),
                               upload_source_files=['experiment.py', 'tcn_utils.py'])

trainer = pl.Trainer(
    max_epochs=CONFIG.epochs,
    gpus=int(torch.cuda.is_available()),
    gradient_clip_val=CONFIG.grad_clip,
    logger=neptune_logger)
trainer.fit(model, train_dataloader, val_dataloader)

model_path = 'tcn.model'
torch.save(model.state_dict(), model_path)
neptune_logger.experiment.log_artifact(model_path)
neptune_logger.experiment.log_artifact(json_states)
neptune_logger.experiment.log_artifact(gmm_model_path)
neptune_logger.experiment.stop()
# %% md [markdown]
# ## Evaluate model

# %%
model.load_state_dict(torch.load('tcn-3.model', map_location=torch.device(device)))

tcn_states = generate_states(model, dataset, 14000)

# %%
tcn_states

# %%
plotting.plot_states(tcn_states, state_numb=dataset.n_states)

# %%
plotting.plot_states(states, state_numb=dataset.n_states)

# %%
stat_metrics.get_KL_divergence_pdf(states, tcn_states)

# %%
tcn_states.eq(torch.Tensor(states)).sum()

# %%
plotting.ts_analysis_plot(tcn_states, 100)

# %%
plotting.ts_analysis_plot(states, 500)

# %%
neptune_logger.experiment.stop()

# %%
