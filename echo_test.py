# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
#sys.path.append("..")
from stat_estimator import *
from traffic_helpers import *
from plot_helpers import *
from timeseries import *
from hmm_helpers import *
from stat_tests import *
from rnn_helpers import *

import seaborn as sns

plt.rcParams['figure.figsize'] = [12, 5]

# %%
pcapfile = '/home/radion/Applications/iot_amazon_echo.pcap'
traffic_dfs = getTrafficFeatures(pcapfile,
                                 #fileIdent='all'
                                 type_of_identifier='MAC',
                                 percentiles=(1,99),
                                 min_samples_to_estimate=100)[0]

# %%
norm_traffic, scalers = normalize_dfs(traffic_dfs, std_scaler=1)
useTrainedGMM = 0
if not useTrainedGMM:
    gmm_models = get_gmm(norm_traffic)
    save_obj(gmm_models, 'echo_gmm')
else:
    gmm_models = load_obj('echo_gmm')
gmm_states = get_mixture_state_predictions(gmm_models, norm_traffic)

# %%
#plot_states_reports(gmm_states, options='t')
#apply_to_dict(ts_acf_plot, gmm_states, lags=200)
#apply_to_dict(plot_acf_df, traffic_dfs, lags=200)
plot_dfs_acf(traffic_dfs, lags=200, saveToFile='amazon_orig')
ts_acfs_dfs(gmm_states, lags=200, saveToFile='amazon_gmm')

# %%
plot_goodput_dfs(traffic_dfs,saveToFile='amazon_orig')

# %%
plot_joint_dfs(traffic_dfs, 'amazon_echo')

#apply_to_dict(plot_joint, traffic_dfs)

# %%
#plot_3D_hist(traffic_dfs, saveToFile='amazon_echo')
    #print(hist)

# %%
plot_states_reports(gmm_states, options='')
apply_to_dict(ts_acf_plot, gmm_states, lags=200)
apply_to_dict(plot_acf_df, traffic_dfs, lags=200)

# %%

# %% [markdown]
# ## HMM test

# %%
hmm_models_fit = apply_to_2dicts(get_hmm_from_gmm_estimate_transitions, gmm_models, norm_traffic)
hmm_states = gener_hmm_states(hmm_models_fit, traffic_dfs)
hmm_dfs = generate_features_from_gmm_states(gmm_models, hmm_states, scalers)

# %%
plot_states_reports(hmm_states, options='', orig_states=gmm_states)
plot_goodput_dfs(hmm_dfs,saveToFile='amazon_hmm')
t = get_KL_divergence(traffic_dfs, hmm_dfs)
plot_dfs_acf(hmm_dfs, lags=200, saveToFile='amazon_hmm')
ts_acfs_dfs(hmm_states, lags=200, saveToFile='amazon_hmm')

# %% [markdown]
# ## RNN test

# %%
window_size = 50
kl = {}
for layers, loss in zip([1,2,3],[0.001, 0.01, 0.02]):
    rnn_models = get_rnn_models(gmm_states, window_size, loss, layers=layers)
    rnn_states = apply_to_2dicts(gener_rnn_states_with_temperature,
                             rnn_models,
                             gmm_states,
                             window_size=window_size,)
                            #temperatures=[1.0,1.7,1.9])

    rnn_dfs = generate_features_from_gmm_states(gmm_models, rnn_states, scalers)
    kl[layers] = get_KL_divergence(traffic_dfs, rnn_dfs)
    plot_dfs_entropies(dfs, smoothing=5, window=50, bar=0, saveToFile='amazon_rnn_layers'+str(layers))
    plot_goodput_dfs(rnn_dfs, saveToFile='amazon_rnn_layers'+str(layers))
    plot_dfs_acf(rnn_dfs, lags=200, saveToFile='amazon_rnn_layers'+str(layers))
    ts_acfs_dfs(rnn_states, lags=200, saveToFile='amazon_rnn_layers' + str(layers))

# %%
plot_dfs_acf(rnn_dfs, lags=200, saveToFile='amazon_rnn')
ts_acfs_dfs(rnn_states, lags=200, saveToFile='amazon_rnn')

# %%
plot_states_reports(rnn_states, options='t')

# %%
plot_joint_dfs(rnn_dfs)

# %%

# %%

# %%
for dfs, filename in zip([traffic_dfs, hmm_dfs, rnn_dfs], ['amazon_orig_50_5','amazon_hmm_50_5','amazon_rnn_50_5']):
    plot_dfs_entropies(dfs, smoothing=5, window=50, bar=0)

# %%
