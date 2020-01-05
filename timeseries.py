import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt

import stat_tests
import traffic_helpers as preproc
import os
import pandas as pd
import scipy
import numpy as np


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
        plt.show()


def ts_acf_plot(y, lags=None, figsize=None, style='bmh'):
    """
        Plot time series, its ACF and PACF

        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig, axes = plt.subplots(nrows=1, ncols=2)  # , figsize=(10, 4))

        # fig = plt.figure()
        # layout = (1, 2)
        # ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        # acf_ax = plt.subplot2grid(layout, (0, 0))
        # pacf_ax = plt.subplot2grid(layout, (0, 1))

        # p_value = sm.tsa.stattools.adfuller(y)[1]
        smt.graphics.plot_acf(y, lags=lags, ax=axes[0])
        smt.graphics.plot_pacf(y, lags=lags, ax=axes[1])
        # plt.tight_layout()
        plt.show()


def ts_acfs_dfs(dfs, lags=200, saveToFile=None, style='bmh'):
    with plt.style.context(style):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
        plot_counter = 0
        for device, direction, y in preproc.iterate_2layer_dict(dfs):
            smt.graphics.plot_acf(y, lags=lags, ax=axes[plot_counter, 0], title='ACF | {}'.format(direction))
            smt.graphics.plot_pacf(y, lags=lags, ax=axes[plot_counter, 1], title='Partial ACF | {}'.format(direction))
            plot_counter += 1
        plt.tight_layout()
        if saveToFile:
            plt.savefig('stat_figures' + os.sep + saveToFile)
        plt.show()


def df_analysis_plot(df, lags=100):
    for parameter in df:
        print(parameter)
        ts_analysis_plot(df[parameter], lags=lags)


def plot_states(states, state_numb=None, figsize=(12, 5)):
    if not state_numb:
        state_numb = len(set(states))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    pd.Series(states).hist(bins=state_numb, ax=axes[0])
    pd.Series(states).plot(style='.', ax=axes[1])
    plt.show()


def plot_series_and_ma(series, ma_window=None, center=True):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    plt.figure()

    if ma_window:
        series.rolling(ma_window, center=center).mean().plot(linewidth=4)  # ,color='g')
    series.plot()
    plt.show()


def plot_states_reports(states, options='ste', orig_states=None):
    if 's' in options:
        preproc.unpack_2layer_traffic_dict(plot_states)(states)
    if 't' in options:
        preproc.unpack_2layer_traffic_dict(ts_analysis_plot)(states, lags=100)
    entropy = preproc.unpack_2layer_traffic_dict(calc_windowed_entropy_discrete)(states)
    print('Entropy stats:\n{}\n'.format(pd.DataFrame(*list(entropy.values())).describe().loc[:, :]))
    if 'e' in options:
        preproc.unpack_2layer_traffic_dict(plot_series_and_ma)(entropy, ma_window=6)
    if orig_states:
        print('KL divergence:\n{}'.format(
            pd.DataFrame(preproc.unpack_2layer_traffic_dict(stat_tests.get_KL_divergence_pmf)(orig_states, states)).T))


def calc_windowed_entropy_discrete(series, window=50):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    state_numb = len(set(series))
    windowed_entropy = []
    for i in range(0, len(series), window):
        series_slice = series.iloc[i:i + window]
        series_pmf = [len(series_slice[series_slice == state]) for state in range(state_numb)]
        windowed_entropy.append(scipy.stats.entropy(series_pmf))

    return pd.Series(windowed_entropy)


def calc_windowed_entropy_cont(series, window=50, kde_bins=1500):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    state_numb = len(set(series))
    windowed_entropy = []
    for i in range(0, len(series), window):
        series_slice = series.iloc[i:i + window]
        if len(series_slice) < 10:
            continue

        series_pdf = scipy.stats.gaussian_kde(series_slice)
        # print(series_slice.values)
        x_values = np.linspace(0, max(series), kde_bins)
        entropy = scipy.stats.entropy(series_pdf(x_values))
        windowed_entropy.append(entropy)
    # print(windowed_entropy)

    return pd.Series(windowed_entropy).fillna(0)
