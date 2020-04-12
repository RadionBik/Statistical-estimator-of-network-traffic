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
import json
import sys
import logging

import numpy as np
import dataclasses

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    out=sys.stdout,
                    format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

# %%
sys.path.append("..")
import mixture_models
import plotting
import pcap_parser
import utils
import stat_metrics
# sys.path.pop()

# %% [markdown]
# ## Initialize parameters

# %%
@dataclasses.dataclass
class ExperimentConfig:
    pcap_file: str = '../traffic_dumps/skypeLANhome.pcap'
    pcap_traffic_kind: utils.TrafficObjects = utils.TrafficObjects.FLOW
    scenario: str = 'skype'
    windows_size: int = 200
    traffic_direction: str = ''
    hidden_size: int = -1
    n_classes: int = -1
    n_levels: int = 7
    kernel_size: int = 4
    epochs: int = 300
    dropout: float = 0.0
    batch_size: int = 16
    optimizer: str = 'Adam'
    learning_rate: float = 0.0005
    grad_clip: float = 1.0
    train_val_splits: int = 10
    log_interval: int = 50


CONFIG = ExperimentConfig()

# %%
traffic_dfs = pcap_parser.get_traffic_features(CONFIG.pcap_file,
                                               type_of_identifier=CONFIG.pcap_traffic_kind,
                                               percentiles=(1, 99),
                                               min_samples_to_estimate=100)[0]

norm_traffic, scalers = utils.normalize_dfs(traffic_dfs, std_scaler=False)

useTrainedGMM = 1
if not useTrainedGMM:
    gmm_models = mixture_models.get_gmm(norm_traffic, sort_components=True)
    utils.save_obj(gmm_models, f'{CONFIG.scenario}_gmm')
else:
    gmm_models = utils.load_obj(f'{CONFIG.scenario}_gmm')

gmm_states = mixture_models.get_mixture_state_predictions(gmm_models, norm_traffic)

# %%
_, _, gmm_model = next(utils.iterate_2layer_dict(gmm_models))
_, direction, states = next(utils.iterate_2layer_dict(gmm_states))
CONFIG.traffic_direction = direction
plotting.plot_gmm_components(gmm_model)

# %%

# %%
json_states = f'gmm_{CONFIG.scenario}_{CONFIG.traffic_direction}.json'
with open(json_states, 'w') as jsf:
    json.dump(states.tolist(), jsf)

# %% [markdown]
# ## Load saved GMM states

# %%
json_states = f'gmm_{CONFIG.scenario}_{CONFIG.traffic_direction}.json'
with open(json_states, 'r') as jsf:
    states = np.array(json.load(jsf))

# %% [markdown]
# ## Initialize model

# %%
import torch
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import neptune

from tcn_utils import train, validate, StatesDataset, generate_states
from tcn_model import TCN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = StatesDataset(states, window=200, device=device)
CONFIG.hidden_size = dataset.n_states
CONFIG.n_classes = dataset.n_states

model = TCN(1, output_size=CONFIG.n_classes,
            num_channels=[CONFIG.hidden_size] * CONFIG.n_levels,
            kernel_size=CONFIG.kernel_size,
            dropout=CONFIG.dropout).to(device)


# %% [markdown]
# ## Training loop

# %% jupyter={"outputs_hidden": true}
NEPTUNE_LOG = 0
optimizer = getattr(optim, CONFIG.optimizer)(model.parameters(), lr=CONFIG.learning_rate)

val_num = len(dataset) // CONFIG.train_val_splits
train_ds, val_ds = random_split(dataset, lengths=(len(dataset) - val_num, val_num))

if NEPTUNE_LOG:
    neptune.init('radion/TCN')
    neptune.create_experiment(name='basic',
                              params=dataclasses.asdict(CONFIG),
                              upload_source_files=['experiment.py', 'tcn_model.py', 'tcn_utils.py'])

for _ in tqdm(range(1, CONFIG.epochs + 1), file=sys.stdout, bar_format='{l_bar}{bar}{r_bar}\n'):
    train(model, optimizer, train_ds, CONFIG, log=NEPTUNE_LOG)
    validate(model, val_ds, CONFIG.n_classes, log=NEPTUNE_LOG)

model_path = 'tcn.model'
torch.save(model.state_dict(), model_path)
if NEPTUNE_LOG:
    neptune.log_artifact(model_path)
    neptune.log_artifact(json_states)
    neptune.stop()

# %% [markdown]
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

# %%
plotting.ts_analysis_plot(tcn_states, 100)

# %%
plotting.ts_analysis_plot(states, 100)

# %%
