import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import random

from traffic_helpers import *

def ts_analysis_plot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title(
            'Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


def plot_states(states, state_numb=None, figsize=(12, 5)):
    if not state_numb:
        state_numb = len(set(states))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    pd.Series(states).hist(bins=state_numb, ax=axes[0])
    pd.Series(states).plot(style='.', ax=axes[1])
    plt.show()


def get_mixture_state_predictions(mixture_model, traffic_dfs):
    mm_pred = construct_dict_2_layers(mixture_model)
    for dev, direct, mm in iterate_traffic_dict(mixture_model):
        mm_pred[dev][direct] = mm.predict(traffic_dfs[dev][direct])
    return mm_pred


def plot_acf_df(df):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    pd.Series([df['IAT'].autocorr(lag)
               for lag in range(round(len(df['pktLen'])))]).plot(ax=axes[0])
    axes[0].xaxis.set_label_text('Lag')
    axes[0].yaxis.set_label_text('ACF, IAT')
    pd.Series([df['pktLen'].autocorr(lag)
               for lag in range(round(len(df['pktLen'])))]).plot(ax=axes[1])
    axes[1].xaxis.set_label_text('Lag')
    axes[1].yaxis.set_label_text('ACF, Packet Size')
    plt.show()


def plot_iat_pktlen(df):
    # plt.figure()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    df['IAT'].plot(ax=axes[0])
    axes[0].xaxis.set_label_text('time, s')
    axes[0].yaxis.set_label_text('IAT, s')
    df['pktLen'].plot(ax=axes[1])
    axes[1].xaxis.set_label_text('time, s')
    axes[1].yaxis.set_label_text('Packet Size, bytes')
    plt.show()

# @access_traffic_df


def plot_goodput(df, resolution='1S', plot=True, saveToFile=None):
    plt.figure()
    # replace indexes with DateTime format
    df.index = pd.to_datetime(df.IAT.cumsum(), unit='s')
    goodput_df = df.resample(resolution).sum()

    ax = (goodput_df['pktLen']/1024).plot(grid=True, lw=3)  # label=direction
    # for {device}')
    ax.set(xlabel='time', ylabel='KB per '+resolution, title=f'Goodput')
    ax.legend()

    if saveToFile:
        plt.savefig('stat_figures'+os.sep+'goodput_' +
                    mod_addr(device)+'_'+saveToFile+'.pdf')


def generate_features_from_gmm_states(gmm_model, states, scalers):

    for _, _, gmm in iterate_traffic_dict(gmm_model):
        gmm

    for _, _, scaler in iterate_traffic_dict(scalers):
        scaler

    gen_samples = np.zeros((len(states), gmm.means_.shape[1]))
    for i, state in enumerate(states.astype('int32')):
        for feature in range(gmm.means_.shape[1]):
            mean = gmm.means_[state][feature]
            var = np.sqrt(gmm.covariances_[state][feature, feature])
            gen_samples[i, feature] = random.gauss(mean, var)

    return pd.DataFrame(scaler.inverse_transform(gen_samples), columns=['IAT', 'pktLen'])

    def get_KL_divergence_value(orig_values, gen_values, x_values):

        kde_orig = scipy.stats.gaussian_kde(orig_values)
        kde_gen = scipy.stats.gaussian_kde(gen_values)
        return scipy.stats.entropy(pd.Series(kde_orig(x_values)), pd.Series(kde_gen(x_values)))


def change_pmf_temperature(preds, temperature=1.0):
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    try:
        probas = np.random.multinomial(1, preds, 1)
    except ValueError:
        probas = np.random.multinomial(1, np.around(preds), 1)
    return np.argmax(probas)


def rnn_gener_state(sample_number_to_gener, init_states, window_size, temperature=1.0):
    start_index = random.randint(0, len(init_states) - window_size - 1)

    seed_states = init_states[start_index: start_index + window_size]
    generated_states = np.zeros(sample_number_to_gener)
    generated_states[:window_size] = seed_states
    print('------ temperature:', temperature)
    for i in range(sample_number_to_gener - window_size):
        # one-hot encode seed_states
        sampled = np.zeros((1, window_size, state_numb))
        for t, state in enumerate(seed_states):
            sampled[0, t, state] = 1
        one_prediction = model.predict(sampled, verbose=0)[0]

        next_state = change_pmf_temperature(one_prediction, temperature)
        generated_states[window_size+i-1] = next_state
        seed_states[0] = next_state
        seed_states = np.roll(seed_states, -1)

    return generated_states


def gener_rnn_states_with_best_temperature(sample_number_to_gener, init_from_states,
                                           window_size):
    least_distance = 1
    for temperature in [1.2, 1.5, 1.8, 2.0]:
        rnn_states = rnn_gener_state(sample_number_to_gener,
                                     init_from_states,
                                     window_size,
                                     temperature)

        min_len = min(len(rnn_states), len(init_from_states))
        distance = get_KL_divergence_value(init_from_states[:min_len],
                                           rnn_states[:min_len],
                                           np.linspace(0, state_numb, 100))
        print(distance)
        if distance < least_distance:
            least_distance = distance
            best_rnn_states = rnn_states

    return best_rnn_states


def get_windowed_training_set_m2m(df, window_size, shift=1):
    feature_number = df.shape[1]
    sample_number = df.shape[0]
    X = np.zeros((sample_number-window_size+1, window_size, feature_number))
    y = np.zeros((sample_number-window_size+1, window_size, feature_number))

    for batch in range(sample_number-window_size+1):
        X[batch, :, :] = df.iloc[batch:batch+window_size, :]
        y[batch, :, :] = df.shift(shift).fillna(0).iloc[batch:batch+window_size, :]
    return X, y
 

def get_batched_training_set_m2m(df, window_size, shift=1):
    batch_num = int(np.ceil(df.shape[0]/window_size))
    zeros_to_append = np.zeros((batch_num*window_size - df.shape[0], df.shape[1]))
    df = df.append(pd.DataFrame(zeros_to_append, columns=df.columns), ignore_index=True)

    X = np.reshape(df.values, (batch_num, window_size, 2)) 
    y = np.reshape(df.shift(shift).fillna(0).values, (batch_num, window_size, 2) )
    
    #return only non-zero batches 
    return X[1:,:,:], y[1:,:,:]


def get_windowed_training_set_m2o(df, window_size, shift=1):
  
    if len(df.shape)==1:
      feature_number = 1
    else:
      feature_number = df.shape[1]

    sample_number = df.shape[0]
    X = np.zeros((sample_number-window_size+1, window_size, feature_number))
    y = np.zeros((sample_number-window_size+1, feature_number))
    print(X.shape)
    if len(df.shape)==1:
      for batch in range(sample_number-window_size):
          X[batch, :, 0] = df.iloc[batch:batch+window_size]
          y[batch, 0] = df.iloc[batch+window_size]
    else:
      for batch in range(sample_number-window_size):
          X[batch, :, :] = df.iloc[batch:batch+window_size, :]
          y[batch, :] = df.iloc[batch+window_size, :]
        
        
    return X, y

def get_one_hot_training_states(states, window_size, step=3):

    sentences = []
    next_states = []
    state_numb = len(set(states))

    for i in range(0, len(states) - window_size, step):
        sentences.append(states[i: i + window_size])
        next_states.append(states[i + window_size])

    print('Number of sequences:', len(sentences))
    print('Unique states:', state_numb)

    #one-hot encoding
    x = np.zeros((len(sentences), window_size, state_numb), dtype=np.bool)
    y = np.zeros((len(sentences), state_numb), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, state in enumerate(sentence):
            x[i, t, state] = 1
        y[i, state] = 1

    return x, y


def plot_training_val(history, metric=None):
    print(history.history.keys())

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    if metric:
        acc = history.history[metric]
        val_acc = history.history['val_'+metric]
        plt.plot(epochs, acc, 'bo', label='Training '+metric)
        plt.plot(epochs, val_acc, 'b', label='Validation '+metric)
        plt.title('Training and validation '+metric)
        plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def calc_windowed_entropy(series, window):
    
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    state_numb = len(set(series))
    windowed_entropy = []
    for i in range(0,len(series),window):
        series_slice = series.iloc[i:i+window]
        series_pmf = [len(series_slice[series_slice==state]) for state in range(state_numb)]
        windowed_entropy.append(entropy(series_pmf))
    
    windowed_entropy = pd.Series(windowed_entropy)
    return windowed_entropy

def plot_series_and_ma(series, ma_window=None, center=True):
    
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    plt.figure()

    if ma_window:
        series.rolling(ma_window, center=center).mean().plot(linewidth=4)#,color='g')
    series.plot()