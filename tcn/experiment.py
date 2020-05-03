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
import dataclasses
import logging
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("..")
import mixture_models
import plotting
import settings
import pcap_parser
import utils
import stat_metrics
from window_estimator import window_estimator
# sys.path.pop()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s %(name)s:%(lineno)d %(levelname)s - %(message)s')

# %% md [markdown]
# ## Initialize parameters

# %%
@dataclasses.dataclass
class ScenarioConfig:
    name: str = 'amazon'
    pcap_file: str = settings.BASE_DIR / 'traffic_dumps/iot_amazon_echo.pcap'
    pcap_traffic_kind: utils.TrafficObjects = utils.TrafficObjects.MAC
    device_identifier: str = '44:65:0d:56:cc:d3'
#     scenario: str = 'skype'
#     pcap_file: str = settings.BASE_DIR / 'traffic_dumps/skypeLANhome.pcap'
#     pcap_traffic_kind: utils.TrafficObjects = utils.TrafficObjects.FLOW
    test_size: int = 0.5
    sort_components: bool = True


SCENARIO_CONFIG = ScenarioConfig()

# %%
traffic_dfs = pcap_parser.get_traffic_features(SCENARIO_CONFIG.pcap_file,
                                               type_of_identifier=SCENARIO_CONFIG.pcap_traffic_kind,
                                               identifiers=[SCENARIO_CONFIG.device_identifier],
                                               percentiles=(1, 99),
                                               min_samples_to_estimate=100)[0]


# %%
train_dfs, test_dfs = utils.split_train_test_dfs(traffic_dfs, SCENARIO_CONFIG.test_size)

plotting.features_acf_dfs(train_dfs)
plotting.features_acf_dfs(test_dfs)

plotting.hist_joint_dfs(train_dfs)
plotting.hist_joint_dfs(test_dfs)

# %%
train_norm_traffic, scalers = utils.normalize_dfs(train_dfs, std_scaler=False)
test_norm_traffic, _ = utils.normalize_dfs(test_dfs, scalers)

# %%
useTrainedGMM = True
gmm_model_name = f'{SCENARIO_CONFIG.name}_gmm'
if not useTrainedGMM:
    gmm_models = mixture_models.get_gmm(train_norm_traffic, sort_components=SCENARIO_CONFIG.sort_components)
    gmm_model_path = utils.save_obj(gmm_models, gmm_model_name)
else:
    gmm_models, gmm_model_path = utils.load_obj(gmm_model_name)

# %%
train_gmm_states = mixture_models.get_mixture_state_predictions(gmm_models, train_norm_traffic)
test_gmm_states = mixture_models.get_mixture_state_predictions(gmm_models, test_norm_traffic)

# %%
def plot_window_estimations(states_dict, **freqs):
    window = utils.construct_dict_2_layers(states_dict)
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 7))
    for index, (dev, direction, st) in enumerate(utils.iterate_2layer_dict(states_dict)):
        window[dev][direction] = window_estimator.plot_report(st, axes=axes[index], **freqs)
    fig.tight_layout()
    return window


LOW_FREQ, HIGH_FREQ = 1 / 512, 1/5

plot_window_estimations(train_gmm_states, low_freq=LOW_FREQ, high_freq=HIGH_FREQ)
plot_window_estimations(test_gmm_states, low_freq=LOW_FREQ, high_freq=HIGH_FREQ)

# %% [markdown] jupyter={"outputs_hidden": true}
# ## HMM

# %%
import markov_models

hmm_models = utils.unpack_2layer_traffic_dict(markov_models.get_hmm_from_gmm_estimate_transitions)(gmm_models, train_norm_traffic)


# %%
def evaluate_states(gen_states, test_states):
    metrics = {}
    metrics['states_ACF_peak'] = window_estimator.plot_report(gen_states, low_freq=LOW_FREQ, high_freq=HIGH_FREQ)
    metrics['states_KL_divergence'] = stat_metrics.get_KL_divergence_pdf(gen_states, test_states)
    plotting.plot_states(gen_states, state_numb=len(set(test_states)))
    pprint(metrics)
    return metrics
    

test_hmm_states = markov_models.gener_hmm_states(hmm_models, test_dfs)
for dev, direct, st in utils.iterate_2layer_dict(test_hmm_states):
    metrics, _ = evaluate_states(st, test_gmm_states[dev][direct])

# %%
test_hmm_dfs = utils.unpack_2layer_traffic_dict(mixture_models.generate_features_from_gmm_states)(gmm_models, test_hmm_states, scalers)
# plotting.features_acf_dfs(test_hmm_dfs)
# plotting.hist_joint_dfs(test_hmm_dfs)
stat_metrics.get_KL_divergence(test_hmm_dfs, test_dfs)
utils.unpack_2layer_traffic_dict(pd.DataFrame.describe)(test_hmm_dfs)

# %% [markdown]
# ## TCN

# %%
# for dev, direction, gmm_model in utils.iterate_2layer_dict(gmm_models):
#     scaler = scalers[dev][direction]
#     train_gmm_state = train_gmm_states[dev][direction]
#     test_gmm_state = test_gmm_states[dev][direction]
#     test_traffic_df = test_dfs[dev][direction]

# plotting.plot_gmm_components(gmm_model)

# %%
_, direction, gmm_model = next(utils.iterate_2layer_dict(gmm_models))
_, _, scaler = next(utils.iterate_2layer_dict(scalers))
_, _, train_gmm_state = next(utils.iterate_2layer_dict(train_gmm_states))
_, _, test_gmm_state = next(utils.iterate_2layer_dict(test_gmm_states))
_, _, test_traffic_df = next(utils.iterate_2layer_dict(test_dfs))

plotting.plot_gmm_components(gmm_model)


# %% md [markdown]
# ## Load saved GMM states

# %%
# json_states = (settings.BASE_DIR / 'tcn' / 'gmm_skype_from.json').as_posix()
# gmm_model_path = (settings.BASE_DIR / 'obj' / 'skype_gmm.pkl').as_posix()
# with open(json_states, 'r') as jsf:
#     states = np.array(json.load(jsf))

# %% md [markdown]
# ## Train model

# %%
import copy

import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import EarlyStopping
import scipy

from base_model import TemporalConvNet
from tcn_utils import StatesDataset, generate_states, get_model_size, get_eff_memory


@dataclasses.dataclass
class NetConfig:
    window_size: int = 100
    traffic_direction: str = ''
    hidden_size: int = -1
    n_classes: int = -1
    n_levels: int = 6
    kernel_size: int = 6
    es_patience: int = 50
    dropout: float = 0.0
    batch_size: int = 64
    optimizer: str = 'Adam'
    learning_rate: float = 0.01
    grad_clip: float = 1.0
    val_size: float = 0.1 
    model_size: int = 0
    effective_memory: int = dataclasses.field(init=False)
    fft_loss: str = ''

    def __post_init__(self):
        self.effective_memory = get_eff_memory(filter_size=self.kernel_size, n_layers=self.n_levels)


NET_CONFIG = NetConfig(traffic_direction=direction,
                       hidden_size=gmm_model.n_components,
                       n_classes=gmm_model.n_components)
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = StatesDataset(train_gmm_state, window=NET_CONFIG.window_size, device=device)
val_num = int(len(dataset) * NET_CONFIG.val_size)
train_dataset, val_dataset = random_split(dataset, lengths=(len(dataset) - val_num, val_num))


pprint(NET_CONFIG)
# %%
# %%
class TCN(LightningModule):
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
        optimizer = getattr(torch.optim, NET_CONFIG.optimizer)(self.parameters(), lr=NET_CONFIG.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)
        return [optimizer], [scheduler]
    
    def _calc_fft_loss(self, output, target):
        if NET_CONFIG.fft_loss == '2D':
            output_max_pool = output.max(dim=2)[0]
            return F.mse_loss(torch.rfft(output_max_pool.float(), signal_ndim=2, normalized=True),
                              torch.rfft(target.float(), signal_ndim=2, normalized=True))
        elif NET_CONFIG.fft_loss == '1D':
            output_max_pool = output.max(dim=2)[0]
            return F.mse_loss(torch.rfft(output_max_pool.float(), signal_ndim=1, normalized=True),
                              torch.rfft(target.float(), signal_ndim=1, normalized=True))

    def _calc_loss(self, output, target, n_classes):
        loss = F.cross_entropy(output.view(-1, n_classes), target.view(-1))
        if NET_CONFIG.fft_loss:
            self.logger.experiment.log_metric('CE_loss', loss)
            fft_loss = self._calc_fft_loss(output, target)
            loss += fft_loss
            self.logger.experiment.log_metric('FFT_loss', fft_loss)
        return loss

    def _calc_accuracy(self, output, target, n_classes):
        pred = output.view(-1, n_classes).data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).sum()
        counter = output.view(-1, n_classes).size(0)
        return 100. * correct / counter

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.unsqueeze(1))
        loss = self._calc_loss(y_hat, y, NET_CONFIG.n_classes)
        accuracy = self._calc_accuracy(y_hat, y, NET_CONFIG.n_classes)

        return {'loss': loss, 'accuracy': accuracy}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        logs = {'log': {'train_loss': avg_loss, 'train_accuracy': avg_accuracy}}
        return logs

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.unsqueeze(1).contiguous())
        loss = self._calc_loss(y_hat, y, NET_CONFIG.n_classes)
        accuracy = self._calc_accuracy(y_hat, y, NET_CONFIG.n_classes)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        generated_states = generate_states(self, val_dataset,
                                           sample_number=1000,
                                           window_size=NET_CONFIG.window_size,
                                           shuffle=False,
                                           prepend_init_states=False,
                                           device=device
                                           )
        try:
            kl_distance = stat_metrics.get_KL_divergence_pdf(train_gmm_state, generated_states)
        except Exception as e:
            logger.error(e)
            kl_distance = np.nan
        ws_distance = scipy.stats.wasserstein_distance(train_gmm_state, generated_states)

        logs = {'val_loss': avg_loss,
                'val_accuracy': avg_accuracy,
                'kl_dist': kl_distance,
                'wasserstein_dist': ws_distance
                }
        logs.update({'log': copy.deepcopy(logs)})
        return logs


model = TCN(1, output_size=NET_CONFIG.n_classes,
            num_channels=[NET_CONFIG.hidden_size] * NET_CONFIG.n_levels,
            kernel_size=NET_CONFIG.kernel_size,
            dropout=NET_CONFIG.dropout)

NET_CONFIG.model_size = get_model_size(model)


train_dataloader = DataLoader(train_dataset,
                              batch_size=NET_CONFIG.batch_size,
                              drop_last=True,
                              shuffle=False)

val_dataloader = DataLoader(val_dataset,
                            batch_size=NET_CONFIG.batch_size,
                            drop_last=True,
                            shuffle=False)


neptune_logger = NeptuneLogger(
    api_key=settings.NEPTUNE_API_TOKEN,
    offline_mode=True,
    close_after_fit=False,
    project_name='radion/TCN',
    experiment_name='train_val?_test_metrics_fft_loss',
    params=dataclasses.asdict(NET_CONFIG),
    upload_source_files=['experiment.py', 'tcn_utils.py']
)


early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=NET_CONFIG.es_patience,
    verbose=False,
    mode='min'
)


trainer = Trainer(
    early_stop_callback=early_stop_callback,
    auto_lr_find=False,
    gpus=int(device=='cuda'),
    gradient_clip_val=NET_CONFIG.grad_clip,
    logger=neptune_logger)
trainer.fit(model, train_dataloader, val_dataloader)

model_path = 'tcn.model'
torch.save(model.state_dict(), model_path)
neptune_logger.experiment.log_artifact(model_path)
neptune_logger.experiment.log_artifact(gmm_model_path)


# evaluation part
tcn_states = generate_states(model, 
                             val_dataset,
                             sample_number=len(test_gmm_state),
                             window_size=NET_CONFIG.window_size,
                             shuffle=False,
                             prepend_init_states=False,
                             device=device)

res_metrics = {}
windows_fig = 'window_est.pdf'
res_metrics['states_ACF_peak'] = window_estimator.plot_report(tcn_states,
                                                              low_freq=LOW_FREQ, high_freq=HIGH_FREQ,
                                                              save_to=windows_fig)
neptune_logger.experiment.log_artifact(windows_fig)
res_metrics['states_KL_divergence'] = stat_metrics.get_KL_divergence_pdf(tcn_states, test_gmm_state)
states_fig = 'gen_states.pdf'
st_fig = plotting.plot_states(tcn_states, state_numb=NET_CONFIG.n_classes)

tcn_features = mixture_models.generate_features_from_gmm_states(gmm_model, tcn_states.numpy(), scaler)
res_metrics['KL_IAT'] = stat_metrics.get_KL_divergence_pdf(test_traffic_df['IAT'], tcn_features['IAT'])
res_metrics['KL_PS'] = stat_metrics.get_KL_divergence_pdf(test_traffic_df['pktLen'], tcn_features['pktLen'])
pprint(res_metrics)
for k, v in res_metrics.items():
    neptune_logger.experiment.log_metric(k, v)

neptune_logger.experiment.stop()
# %%
window_estimator.plot_report(tcn_states, low_freq=LOW_FREQ, high_freq=HIGH_FREQ, save_to=windows_fig)
# %% md [markdown]
# ## Evaluate model

# %%
# model.load_state_dict(torch.load('tcn-36.model', map_location=torch.device(device)))

# %%
neptune_logger.experiment.stop()

# %%

# %%
plotting.plot_states(tcn_states, state_numb=NET_CONFIG.n_classes)

# %%
plotting.plot_states(test_gmm_state, state_numb=len(set(test_gmm_state)))


# %%
plotting.ts_analysis_plot(tcn_states[:3000], 500)


# %%
