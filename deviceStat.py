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

# %% [markdown]
# plan of actions:
# 1. extract traffic for Amazon Echo 44:65:0d:56:cc:d3
# 2. variables to check: 
#     * loss
#     * temperature
#     * window_size

# %%

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
#sys.path.append("..")
import stat_estimator as estimator
import plotting

# %%
import traffic_helpers as preprocessors
import hmm_helpers as mm_utils
import ts_helpers as timeseries 

# %%
import stat_tests as tests
import rnn_helpers as rnn_utils
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 4]

# %%
pcapfile = 'traffic_dumps/skypeLANhome.pcap'
traffic_dfs = estimator.getTrafficFeatures(pcapfile, 
                                           typeIdent='flow',
                                           fileIdent='all',
                                           percentiles=(1,99),
                                           min_samples_to_estimate=100)[0]

# %% [markdown]
# ## Original traffic
#

# %%

# %%
norm_traffic, scalers = preprocessors.normalize_dfs(traffic_dfs, std_scaler=1)

useTrainedGMM = 0
if not useTrainedGMM:
    gmm_models = mm_utils.get_gmm(norm_traffic)
    preprocessors.save_obj(gmm_models, 'skype_gmm')
else:
    gmm_models = preprocessors.load_obj('skype_gmm')
    
gmm_states = rnn_utils.get_mixture_state_predictions(gmm_models, norm_traffic)

# %%

# %%
# examples for other plot utils
# plotting.hist_2d_dfs(traffic_dfs, states=gmm_states)
# plotting.hist_dfs(traffic_dfs, logScale=False)
# plotting.hist_3d(traffic_dfs, save_to='hist3d_skype.pdf')
# preprocessors.unpack_2layer_traffic_dict(timeseries.df_analysis_plot)(traffic_dfs, lags=400)

# %% jupyter={"outputs_hidden": true}
plotting.hist_joint_dfs(traffic_dfs)#, save_to='joint2d_skype.pdf')
plotting.goodput_dfs(traffic_dfs)
plotting.plot_dfs_acf(traffic_dfs, lags=500)#, saveToFile='acf_df_skype_orig.pdf')

# %% jupyter={"outputs_hidden": true}
timeseries.ts_acfs_dfs(gmm_states,lags=300)#, saveToFile='skype_gmm')
timeseries.plot_states_reports(gmm_states, options='ste')

# %% [markdown]
# ## Hidden Markov Model

# %%
#hmm_models_calc = apply_to_2dicts(get_hmm_from_gmm_pred_calc_transitions, gmm_models, gmm_pred_states)
hmm_models_fit = preprocessors.unpack_2layer_traffic_dict(mm_utils.get_hmm_from_gmm_estimate_transitions)(gmm_models, norm_traffic)
hmm_states = mm_utils.gener_hmm_states(hmm_models_fit, traffic_dfs)
hmm_dfs = mm_utils.generate_features_from_gmm_states(gmm_models, hmm_states, scalers)

# %% jupyter={"outputs_hidden": true}
plotting.hist_joint_dfs(hmm_dfs)
plotting.goodput_dfs(hmm_dfs)
plotting.plot_dfs_acf(hmm_dfs, lags=500)

# %% jupyter={"outputs_hidden": true}
timeseries.ts_acfs_dfs(hmm_states,lags=300)
timeseries.plot_states_reports(hmm_states, options='ste', orig_states=gmm_states)

# %%
t = tests.get_KL_divergence(traffic_dfs, hmm_dfs)
t = tests.get_ks_2sample_test(traffic_dfs, hmm_dfs)

# %%

# %% [markdown]
# ## RNN as classifier

# %%

# %%
rnn_models = rnn_utils.get_rnn_models(gmm_states, 200, 0.04)

# %%
rnn_states = preprocessors.unpack_2layer_traffic_dict(rnn_utils.gener_rnn_states_with_temperature)(rnn_models, gmm_states, window_size=200)

# %%
rnn_dfs = mm_utils.generate_features_from_gmm_states(gmm_models, rnn_states, scalers)

# %%
timeseries.ts_acfs_dfs(rnn_states,lags=300)
timeseries.plot_states_reports(rnn_states, options='ste')

# %%

# %% jupyter={"outputs_hidden": true}
plotting.hist_joint_dfs(rnn_dfs)
plotting.goodput_dfs(rnn_dfs)
plotting.plot_dfs_acf(rnn_dfs, lags=500)

# %%
t = tests.get_KL_divergence(traffic_dfs, rnn_dfs)
t = tests.get_ks_2sample_test(traffic_dfs, rnn_dfs)

# %%
for dfs in [traffic_dfs, hmm_dfs, rnn_dfs]:
    plotting.entropies_dfs(dfs, smoothing=5, window=50, bar=0)

t=tests.get_KL_divergence(traffic_dfs, hmm_dfs)
t=tests.get_KL_divergence(traffic_dfs, rnn_dfs)


# %%
def test_rnn_by_window_size(window_range, traffic_dfs, gmm_states, gmm_models, scalers, sample_number_to_gener, temperatures=None, loss_threshold = 0.04 ):
    entropies = {}
    models = {}
    kl_div = {}
    dfs_dict = {}
    for window_size in window_range:
        print(f'\n--------------------Window size is {window_size}-----------------------')

                   
        rnn_models = get_rnn_models(gmm_states, window_size, loss_threshold)

        
        models['window_{}'.format(window_size)] = rnn_models
        
        rnn_states = apply_to_2dicts(gener_rnn_states_with_temperature,
                             rnn_models,
                             gmm_states,
                             sample_number_to_gener=sample_number_to_gener, 
                             window_size=window_size)

        rnn_dfs = generate_features_from_gmm_states(gmm_models, rnn_states, scalers)
        dfs_dict['window_{}'.format(window_size)] = rnn_dfs
        entropies['window_{}'.format(window_size)] = plot_dfs_entropies(rnn_dfs, smoothing=5, window=20, bar=0)
        kl_div['window_{}'.format(window_size)] = get_KL_divergence(traffic_dfs, rnn_dfs)
        #plot_states_reports(gen_states, options='', orig_states=states)
        #print('Entropy rate is {:.2f}'.format(entropies[window_size]))
        
    return models, entropies, kl_div, dfs_dict

# %%
models_trial = {}
entropies_trial = {} 
kl_div_trial = {}
rnn_dfs_trial = {}
for trial in range(3):
    print('_______________testing TRIAL={}_____________________________________________________'.format(trial))
    models, entropies, kl_div, dfs_dict = test_rnn_by_window_size([200, 250, 300], 
                                                                  traffic_dfs, 
                                                                  gmm_states, 
                                                                  gmm_models, 
                                                                  scalers, 
                                                                  sample_number_to_gener, 
                                                                  loss_threshold=0.045)
    models_trial['trial_{}'.format(trial)] = models
    entropies_trial['trial_{}'.format(trial)] = entropies
    kl_div_trial['trial_{}'.format(trial)] = kl_div
    rnn_dfs_trial['trial_{}'.format(trial)] = dfs_dict


# %%

# %%

# %%
def decorr_df(df, shift=1):
     return df-pd.Series(df).shift(shift).fillna(0)
    
def corr_df(df, shift=1):
    return df+pd.Series(df).shift(-shift).fillna(0)

# %% [markdown]
#

# %%

# %%

# %%
