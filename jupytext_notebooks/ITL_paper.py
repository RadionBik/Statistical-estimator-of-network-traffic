# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: notebooks_jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   notebooks_jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.1
# ---

# %%

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
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


# %%
# plt.rcParams['figure.figsize'] = [10, 10]

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
IS_SKYPE = 1
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
                            rnn_stop_loss=0.02,
                            rnn_hidden_layers=2,
                            pcap_file='../traffic_dumps/iot_amazon_echo.pcap',
                            pcap_identifier=utils.TrafficObjects.MAC,
                            device_identifier='../addresses_to_check.txt')

# %%
traffic_dfs = pcap_parser.get_traffic_features(CONFIG.pcap_file,
                                               type_of_identifier=CONFIG.pcap_identifier,
                                               file_with_identifiers=CONFIG.device_identifier,
                                               percentiles=(1, 99),
                                               min_samples_to_estimate=100)[0]

# %% [markdown]
# ## Original traffic
#

# %%

# %%
norm_traffic, scalers = utils.normalize_dfs(traffic_dfs, std_scaler=1)

useTrainedGMM = 0
if not useTrainedGMM:
    gmm_models = mixture_models.get_gmm(norm_traffic)
    utils.save_obj(gmm_models, f'{CONFIG.scenario}_gmm')
else:
    gmm_models = utils.load_obj(f'{CONFIG.scenario}_gmm')

gmm_states = rnn_utils.get_mixture_state_predictions(gmm_models, norm_traffic)

# %%

# %%
# examples for other plot utils
# plotting.hist_2d_dfs(traffic_dfs, states=gmm_states)
# plotting.hist_dfs(traffic_dfs, logScale=False)
# plotting.hist_3d(traffic_dfs, save_to='hist3d_skype.pdf')
# preprocessors.unpack_2layer_traffic_dict(timeseries.df_analysis_plot)(traffic_dfs, lags=400)

# %%
plotting.hist_joint_dfs(traffic_dfs, save_to=f'joint2d_{CONFIG.scenario}.pdf')
plotting.goodput_dfs(traffic_dfs)
plotting.features_acf_dfs(traffic_dfs, lags=500)

# %% [markdown]
# RNN_WINDOW_SIZE was selected according to ACF of the states below 

# %%
plotting.ts_acfs_dfs(gmm_states, lags=300) 
plotting.plot_states_reports(gmm_states, options='ste')

# %% [markdown]
# ## Hidden Markov Model

# %%
hmm_models_fit = utils.unpack_2layer_traffic_dict(markov_models.get_hmm_from_gmm_estimate_transitions)(gmm_models,
                                                                                                       norm_traffic)
hmm_states = markov_models.gener_hmm_states(hmm_models_fit, traffic_dfs)
hmm_dfs = mixture_models.generate_features_from_gmm_states(gmm_models, hmm_states, scalers)

# %%
plotting.hist_joint_dfs(hmm_dfs)
plotting.goodput_dfs(hmm_dfs)
plotting.features_acf_dfs(hmm_dfs, lags=500)

# %%
plotting.ts_acfs_dfs(hmm_states, lags=300)
plotting.plot_states_reports(hmm_states, options='ste', orig_states=gmm_states)

# %%
t = stat_metrics.get_KL_divergence(traffic_dfs, hmm_dfs)
t = stat_metrics.get_ks_2sample_test(traffic_dfs, hmm_dfs)

# %%

# %% [markdown]
# ## RNN as classifier

# %%
from tensorflow import keras
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential


@utils.unpack_2layer_traffic_dict
def get_rnn_models(train_states, window_size, loss_threshold, layers=1):
    stop_callback = rnn_utils.StopAtLossValue(loss_threshold)
    # stop_callback = EarlyStopping(patience=5,
    #                               restore_best_weights=True,
    #                               monitor='val_loss')
    state_numb = int(max(set(train_states)) + 1)
    X, y = rnn_utils.get_one_hot_training_states(train_states, window_size, step=5)
    model = Sequential()
    for _ in range(layers - 1):
        model.add(GRU(state_numb,
                      activation='relu',
                      input_shape=(window_size, state_numb),
                      return_sequences=True))

    model.add(GRU(state_numb, input_shape=(window_size, state_numb)))
    model.add(Dropout(0.2))
    model.add(Dense(state_numb, activation='softmax'))

    optimizer = keras.optimizers.RMSprop(lr=0.005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer)

    history = model.fit(X,
                        y,
                        epochs=100,
                        validation_split=0.2,
                        batch_size=20,
                        callbacks=[stop_callback])

    return model


rnn_models = get_rnn_models(gmm_states,
                            CONFIG.rnn_window_size,
                            CONFIG.rnn_stop_loss,
                            layers=CONFIG.rnn_hidden_layers)


# %%
@utils.unpack_2layer_traffic_dict
def gener_rnn_states_with_temperature(model,
                                      orig_states,
                                      window_size,
                                      temperatures=None):
    import numpy as np

    if not temperatures:
        # this is the place for experiments
        init_entropy = np.mean(stat_metrics.calc_windowed_entropy_discrete(orig_states[:window_size]))
        temperatures = [init_entropy]
        if init_entropy < 1.7:
            temperatures += [init_entropy - 0.2, init_entropy + 0.2]
        else:
            temperatures += [init_entropy - 0.4, init_entropy - 0.2]
        print('\nSelected base temperature from init_states: {:.3f}'.format(init_entropy))

    temper_results = {}
    for temperature in temperatures:
        print('Trying temperature={:.3f}...'.format(temperature))
        rnn_states = rnn_utils.rnn_gener_state(model,
                                               full_init_states=orig_states,
                                               window_size=window_size,
                                               temperature=temperature)

        distance = stat_metrics.get_KL_divergence_pmf(orig_states, rnn_states)
        temper_results[temperature] = {'distance': distance,
                                       'states': rnn_states}

        print('Got KL={:.3f}'.format(distance))

    best_temp, best_one = sorted(temper_results.items(), key=lambda x: x[1]['distance'])[0]
    print('\nBest KL divergence={:.3f} for states with temperature={:.3f}'.format(best_one['distance'],
                                                                                  best_temp))

    return best_one['states']


rnn_states = gener_rnn_states_with_temperature(rnn_models,
                                               gmm_states,
                                               window_size=CONFIG.rnn_window_size,
                                               temperatures=(1.7,))

# %%
rnn_dfs = mixture_models.generate_features_from_gmm_states(gmm_models, rnn_states, scalers)

# %%
plotting.ts_acfs_dfs(rnn_states, lags=300)
plotting.plot_states_reports(rnn_states, options='ste')

# %%
plotting.hist_joint_dfs(rnn_dfs)
plotting.goodput_dfs(rnn_dfs)
plotting.features_acf_dfs(rnn_dfs, lags=500)

# %%
t = stat_metrics.get_KL_divergence(traffic_dfs, rnn_dfs)
t = stat_metrics.get_ks_2sample_test(traffic_dfs, rnn_dfs)

# %% [markdown]
# ## Rolling entropies

# %%
for dfs in [traffic_dfs, hmm_dfs, rnn_dfs]:
    plotting.entropies_dfs(dfs, smoothing=5, window=50, bar=0)

# %%

# %%
