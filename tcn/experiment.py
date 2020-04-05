# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys

sys.path.append("..")
from dataclasses import dataclass
import mixture_models
import markov_models
import plotting
import rnn_utils
import pcap_parser
import utils
import stat_metrics
import numpy as np
sys.path.pop()


# %%
@dataclass
class ScenarioConfig:
    scenario: str
    rnn_window_size: int
    rnn_stop_loss: float
    rnn_hidden_layers: int
    pcap_file: str
    pcap_identifier: utils.TrafficObjects
    device_identifier: str = None


# %%
IS_SKYPE = True
if IS_SKYPE:
    CONFIG = ScenarioConfig(scenario='skype',
                            rnn_window_size=200,
                            rnn_stop_loss=0.04,
                            rnn_hidden_layers=1,
                            pcap_file='../traffic_dumps/skypeLANhome.pcap',
                            pcap_identifier=utils.TrafficObjects.FLOW)

else:
    CONFIG = ScenarioConfig(scenario='amazon',
                            rnn_window_size=50,
                            rnn_stop_loss=0.04,
                            rnn_hidden_layers=1,
                            pcap_file='../traffic_dumps/iot_amazon_echo.pcap',
                            pcap_identifier=utils.TrafficObjects.MAC,
                            device_identifier='../addresses_to_check.txt')

# %%
traffic_dfs = pcap_parser.get_traffic_features(CONFIG.pcap_file,
                                               type_of_identifier=CONFIG.pcap_identifier,
                                               file_with_identifiers=CONFIG.device_identifier,
                                               percentiles=(1, 99),
                                               min_samples_to_estimate=100)[0]

# %%
norm_traffic, scalers = utils.normalize_dfs(traffic_dfs, std_scaler=False)

useTrainedGMM = 1
if not useTrainedGMM:
    gmm_models = mixture_models.get_gmm(norm_traffic, sort_components=True)
    utils.save_obj(gmm_models, f'{CONFIG.scenario}_gmm')
else:
    gmm_models = utils.load_obj(f'{CONFIG.scenario}_gmm')

gmm_states = rnn_utils.get_mixture_state_predictions(gmm_models, norm_traffic)

# %%
_, _, gmm_model = next(utils.iterate_2layer_dict(gmm_models))
_, _, states = next(utils.iterate_2layer_dict(gmm_states))

# %%
plotting.plot_gmm_components(gmm_model)

# %%
sys.path

# %%
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from tcn_utils import train, evaluate, StatesDataset
from tcn_model import TCN


dataset = StatesDataset(states, window=CONFIG.rnn_window_size)

parameters = {
    'hidden_size': dataset.n_states,
    'n_levels': 7,
    'n_classes': dataset.n_states,
    'kernel_size': 4,
    'epochs': 300,
    'dropout': 0.0,
    'learning_rate': 0.0005,
    'batch_size': 16,
    'optimizer': 'RMSprop',
    'grad_clip': 1.0,
    'log_interval': 50,
    'train_val_splits': 10,
}

val_num = len(dataset) // parameters['train_val_splits']
train_ds, val_ds = random_split(dataset, lengths=(len(dataset) - val_num, val_num))

channel_sizes = [parameters['hidden_size']] * parameters['n_levels']

model = TCN(1, parameters['n_classes'], channel_sizes, parameters['kernel_size'], dropout=parameters['dropout'])

lr = parameters['learning_rate']
optimizer = getattr(optim, parameters['optimizer'])(model.parameters(), lr=lr)


for ep in tqdm(range(1, parameters['epochs'] + 1), file=sys.stdout, bar_format='{l_bar}{bar}{r_bar}\n'):
    train(model, optimizer, train_ds, ep, parameters)
    evaluate(model, val_ds, parameters['n_classes'])

# %%
