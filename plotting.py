from statsmodels import api as sm
from statsmodels.tsa import api as smt

import mixture_models
import stat_metrics
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from collections import defaultdict

import utils
import settings
from stat_metrics import calc_windowed_entropy_discrete, calc_smoothed_entropies_dfs

FIG_PARAMS = {'low_iat': 0.0000001,
              'high_iat': 200,
              'high_pktlen': 1500,
              'bins': 100}

RESULT_FOLDER = f'{settings.BASE_DIR}{os.sep}figures'


def quantiles_acf_dfs(traffic_dfs, save_to=None):
    """
    draws statistical figures for a traffic DataFrame, includes BoxPlot and Autocorrelation
    for PktLen and IAT
    """

    for device, direction, df in utils.iterate_2layer_dict(traffic_dfs):
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        fig.suptitle('{} {}'.format(direction, device))
        pd.Series([df['pktLen'].autocorr(lag) for lag in range(round(len(df['pktLen']) / 2))]).plot(ax=axes[0, 1],
                                                                                                    grid=True)
        axes[0, 1].xaxis.set_label_text('Lag')
        df['pktLen'].plot.box(ax=axes[0, 0], vert=False, grid=True)
        axes[0, 0].xaxis.set_label_text('bytes')
        axes[0, 1].yaxis.set_label_text('Autocorrelation')
        pd.Series([df['IAT'].autocorr(lag) for lag in range(round(len(df['IAT']) / 2))]).plot(ax=axes[1, 1], grid=True)
        df['IAT'].plot.box(ax=axes[1, 0], vert=False, grid=True)
        axes[1, 0].xaxis.set_label_text('seconds')
        axes[1, 1].yaxis.set_label_text('Autocorrelation')
        axes[1, 1].xaxis.set_label_text('Lag')
        if save_to:
            plt.savefig(RESULT_FOLDER + os.sep + 'stat_props_' +
                        utils.mod_addr(device) + '_' + direction + '_' + save_to + '.pdf')


def hist_2d_dfs(traffic_dfs, parameters=('pktLen', 'IAT'), log_scale=False, states=None, save_to=None):
    """
    plot2D() vizualizes 'IAT' and 'pktLen' parameters as a scatter plot
    """
    for device in traffic_dfs:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        for dir_numb, direction in enumerate(traffic_dfs[device]):
            df = traffic_dfs[device][direction]
            # fig = plt.figure()
            # ax[dir_numb] = plt.gca()
            if states:
                colors = states[device][direction]
            else:
                colors = 'b'
            ax[dir_numb].scatter(df[parameters[1]], df[parameters[0]], c=colors, label=f"direction '{direction}'",
                                 alpha=0.2)  # , markeredgecolor='none', )
            if log_scale:
                ax[dir_numb].set_xscale('log')
            ax[dir_numb].set_xlabel('IAT, s')
            ax[dir_numb].set_ylabel('Payload size, bytes')
            ax[dir_numb].legend(loc='best')
            ax[dir_numb].grid(True)
            ax[dir_numb].set_title(device)

        if save_to:
            plt.savefig(RESULT_FOLDER + os.sep + 'hist2d_' +
                        utils.mod_addr(device) + '_' + save_to + '.pdf')


def hist_dfs(traffic, log_scale=True, save_to=None):
    """
    hist_dfs() plots histograms from the layered dict with packet features.
    'logScale' affects only IAT parameter.
    'saveToFile' suppresses output and saves plots to the disk,
    (to exclude outliers)
    """
    with plt.style.context('bmh'):
        print('Preparing histograms for the traffic...')
        for device in traffic:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
            fig.suptitle('Histogram for ' + device, fontsize=16)
            ax = ax.flatten()
            dev_fig_props = define_figure_properties_per_device(
                traffic[device], log_scale)

            for direction in traffic[device]:
                # create a plain figure
                param_list = ['pktLen', 'IAT']
                for parameter in param_list:

                    if parameter == 'ts':
                        continue
                    deviceData = traffic[device][direction][parameter]

                    plotNumb = dev_fig_props[parameter]['plot_numb']
                    if direction == 'to':
                        plotNumb = plotNumb + 2

                    # plot histogram
                    ax[plotNumb].hist(deviceData, bins=dev_fig_props[parameter]['bins'],
                                      range=dev_fig_props[parameter]['range'], density=True)

                    # Annotate diagram
                    ax[plotNumb].set_xlabel(dev_fig_props[parameter]['x_subscr'])

                    if (parameter == 'IAT') and log_scale:
                        ax[plotNumb].set_xscale("log")
                    ax[plotNumb].set_title('direction: {} ({} packets)'.format(direction, len(deviceData)))
                    ax[plotNumb].grid(True)
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.88)
            if save_to:
                plt.savefig(RESULT_FOLDER + os.sep + 'hist_' +
                            utils.mod_addr(device) + '_' + save_to + '.pdf')
                # else:
        plt.show()
    # plt.pause(0.001)
    # input("Press any key to continue.")


def plot_hist_kde_em(traffic, kde_estimators, em_estimators=None, log_scale=True, save_to_file=False,
                     min_samples_to_estimate=15):
    """
    plot_hist_kde_em() plots histograms, KDEs and EM estimations from the
    layered dict with packet features.
    'logScale' affects only IAT parameter.
    'saveToFile' suppresses output and saves plots to the disk,
    'percentile' defines the percentile above which the data should be omitted
    (to exclude outliers)
    """

    print('Preparing plots for the estimations...')
    x_values = get_x_values_dict(traffic, log_scale)

    kde_values = mixture_models.get_KDE_values(kde_estimators, x_values)

    # em_values = get_EM_values(em_estimators)
    em_values = mixture_models.get_EM_values_dict(em_estimators, x_values)

    for device in traffic:

        if traffic[device]['from']['IAT'].shape[0] < min_samples_to_estimate or \
                traffic[device]['to']['IAT'].shape[0] < min_samples_to_estimate:
            row_numb = 1
        else:
            row_numb = 2

        fig, ax = plt.subplots(nrows=row_numb, ncols=2, figsize=[12, 5])
        fig.suptitle('Estimations for ' + device, fontsize=16)
        ax = ax.flatten()
        dev_fig_props = define_figure_properties_per_device(
            traffic[device], log_scale)

        for direction in traffic[device]:
            # create a plain figure
            param_list = ['pktLen', 'IAT']
            for parameter in param_list:

                if parameter == 'ts':
                    continue
                deviceData = traffic[device][direction][parameter]
                if len(deviceData) < min_samples_to_estimate:
                    continue

                plotNumb = dev_fig_props[parameter]['plot_numb']
                if direction == 'to':
                    plotNumb = plotNumb + 2
                    if row_numb == 1:
                        plotNumb = plotNumb - 2

                x = x_values[device][direction][parameter][1:-1]

                # plot histogram
                ax[plotNumb].hist(deviceData, bins=dev_fig_props[parameter]['bins'],
                                  range=dev_fig_props[parameter]['range'], density=True)

                if kde_values[device][direction][parameter] is not None:
                    ax[plotNumb].plot(x, kde_values[device][direction][parameter][1:-1], color="red", lw=1.5,
                                      label='KDE', linestyle='--')

                if em_values[device][direction][parameter] is not None:
                    ax[plotNumb].plot(x, em_values[device][direction][parameter][1:-1], color="black", lw=1.5,
                                      label='EM', linestyle='--')

                ax[plotNumb].legend()

                # Annotate diagram
                ax[plotNumb].set_xlabel(dev_fig_props[parameter]['x_subscr'])

                if (parameter == 'IAT') and log_scale:
                    ax[plotNumb].set_xscale("log")
                ax[plotNumb].set_title('direction: {} ({} packets)'.format(direction, len(deviceData)))
                ax[plotNumb].grid(True)
                # ax[plotNumb].set_ylim([0, max(max(kde_values[device][direction][parameter]),
                # max(em_values[device][direction][parameter]))])
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)
        if save_to_file:
            figure_name = f'figures{os.sep}hist{utils.mod_addr(device).replace(" ", "_")}_.svg'
            print('Saved figure as {}'.format(figure_name))
            plt.savefig(figure_name)
            # else:
    plt.show()
    # plt.pause(0.001)
    # input("Press any key to continue.")


def getXAxisValues(parameter=None, logScale=False, max_param=None, traffic=None, precision=1000):
    if max_param:
        upper_bound = max_param
    elif isinstance(traffic, pd.Series):
        upper_bound = max(traffic)
    else:
        upper_bound = 200

    if not logScale:
        x = np.linspace(0, upper_bound, precision)

    if (parameter == 'IAT') and logScale:
        x = np.logspace(np.log10(0.000001), upper_bound, precision)

    return x


def get_x_values_dict(traffic, logScale=False):
    x_values = utils.construct_new_dict_no_ts(traffic)
    max_params = _find_max_parameters(traffic)
    for device in traffic:
        for direction in traffic[device]:
            for parameter in ['pktLen', 'IAT']:
                # if parameter == 'ts':
                #    continue

                x_values[device][direction][parameter] = getXAxisValues(
                    parameter=parameter, traffic=traffic[device][direction][parameter], max_param=max_params[parameter],
                    logScale=logScale)

    return x_values


def define_figure_properties_per_device(device_traffic, logScale=True):
    device_props = {'IAT': {}, 'pktLen': {}}
    max_value = defaultdict(dict)
    for direction in device_traffic:
        for parameter in ['pktLen', 'IAT']:

            try:
                max_value[parameter][direction] = max(device_traffic[direction][parameter])
            except ValueError:
                max_value[parameter][direction] = 0

            if parameter == 'pktLen':

                try:
                    maxParam = FIG_PARAMS['high_pktlen'] if not device_traffic[direction][parameter] else \
                        max(max_value[parameter].values()) + 50
                except ValueError:
                    maxParam = FIG_PARAMS['high_pktlen'] if device_traffic[direction][parameter].empty else \
                        max(max_value[parameter].values()) + 50

                device_props[parameter]['range'] = (0, maxParam)
                device_props[parameter]['x_subscr'] = 'Payload Length, bytes'
                device_props[parameter]['bins'] = np.linspace(
                    0, maxParam, FIG_PARAMS['bins'])
                device_props[parameter]['plot_numb'] = 0

            elif parameter == 'IAT':

                try:
                    maxParam = FIG_PARAMS['high_iat'] if not device_traffic[direction][parameter] else \
                        max(max_value[parameter].values())
                except ValueError:
                    maxParam = FIG_PARAMS['high_iat'] if device_traffic[direction][parameter].empty else \
                        max(max_value[parameter].values())

                device_props[parameter]['x_subscr'] = 'IAT, s'
                device_props[parameter]['plot_numb'] = 1

                device_props[parameter]['range'] = (0, maxParam)

                if logScale:
                    device_props[parameter]['bins'] = np.logspace(
                        np.log10(FIG_PARAMS['low_iat']), np.log10(maxParam), FIG_PARAMS['bins'])
                else:
                    device_props[parameter]['bins'] = np.linspace(
                        0, maxParam, FIG_PARAMS['bins'])
            else:
                continue

    # print(device_props)
    return device_props


def defineFigureProperties(parameter, logScale, traffic=None, percentile=100):
    fig_prop = {}
    # hist_prop = {}
    if parameter == 'pktLen':
        fig_prop['param'] = 'Payload Length'
        fig_prop['unit'] = ", bytes"
        fig_prop['range'] = (0, 1500)
        fig_prop['bins'] = np.linspace(0, 1500, FIG_PARAMS['bins'])
        fig_prop['plot_numb'] = 0
    elif parameter == 'IAT':
        fig_prop['param'] = 'IAT'
        fig_prop['unit'] = ", s"
        fig_prop['range'] = (0, FIG_PARAMS['high_iat']
        if not traffic else np.percentile(traffic, percentile))
        fig_prop['plot_numb'] = 1

        if logScale:
            fig_prop['bins'] = np.logspace(np.log10(FIG_PARAMS['low_iat']), np.log10(
                FIG_PARAMS['high_iat'] if not traffic else np.percentile(traffic, percentile)), FIG_PARAMS['bins'])

        else:
            fig_prop['bins'] = np.linspace(0, FIG_PARAMS['high_iat'] if not traffic else np.percentile(
                traffic, percentile), FIG_PARAMS['bins'])

    return fig_prop


def features_acf_dfs(dfs, lags=500, save_to=None):
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=[14, 7])
    plot_counter = 0
    for device, direction, df in utils.iterate_2layer_dict_copy(dfs):
        df.index = [i for i in range(df.shape[0])]
        df['pktLen'].plot(ax=ax[plot_counter, 0], grid=1)
        ax[plot_counter, 0].yaxis.set_label_text('PS, bytes')
        ax[plot_counter, 0].set_title('{} {}'.format(direction, device))

        pd.Series([df['pktLen'].autocorr(lag)
                   for lag in range(lags)]).plot(ax=ax[plot_counter, 1], grid=1)
        ax[plot_counter, 1].yaxis.set_label_text('ACF, PS')
        ax[plot_counter, 1].xaxis.set_label_text('Lag')

        # smt.graphics.plot_acf(df['pktLen'], lags=lags, ax=axes[1])
        df['IAT'].plot(ax=ax[plot_counter + 1, 0], grid=1)
        ax[plot_counter + 1, 0].yaxis.set_label_text('IAT, s')

        pd.Series([df['IAT'].autocorr(lag)
                   for lag in range(lags)]).plot(ax=ax[plot_counter + 1, 1], grid=1)
        ax[plot_counter + 1, 1].yaxis.set_label_text('ACF, IAT')
        ax[plot_counter + 1, 1].xaxis.set_label_text('Lag')

        plot_counter += 2

    plt.tight_layout()
    if save_to:
        plt.savefig(RESULT_FOLDER + os.sep + save_to)

    plt.show()


def goodput(df, resolution='1S', plot=True, save_to=None):
    plt.figure()
    # replace indexes with DateTime format
    df.index = pd.to_datetime(df.IAT.cumsum(), unit='s')
    goodput_df = df.resample(resolution).sum()

    ax = (goodput_df['pktLen'] / 1024).plot(grid=True, lw=3)  # label=direction
    # for {device}')
    ax.set(xlabel='time', ylabel=f'KB per {resolution}', title=f'Goodput')
    # ax.legend()
    plt.show()
    if save_to:
        plt.savefig(RESULT_FOLDER + os.sep + save_to)


def goodput_dfs(dfs, resolution='1S', save_to=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[12, 5])
    plot_counter = 0
    for device, direction, df in utils.iterate_2layer_dict_copy(dfs):
        # replace indexes with DateTime format
        df.index = pd.to_datetime(df.IAT.cumsum(), unit='s')
        goodput_df = df.resample(resolution).sum()

        (goodput_df['pktLen'] / 1024).plot(grid=True, lw=3, ax=ax[plot_counter])  # label=direction
        ax[plot_counter].set(xlabel='time',
                             ylabel='Goodput kB/' + resolution,
                             title=direction + ' ' + device)
        # ax.legend()
        plot_counter += 1
    if save_to:
        plt.savefig(RESULT_FOLDER + os.sep + save_to)

    plt.show()


def entropies_dfs(traffic_dfs,
                  smoothing=10,
                  kde_bins=500,
                  window=50,
                  save_to=None,
                  title=None):

    plt.figure()
    smoothed_entropies = calc_smoothed_entropies_dfs(traffic_dfs, smoothing, window, kde_bins)

    ax = smoothed_entropies.plot(grid=True, figsize=(8, 3))

    if not title:
        title = 'Rolling entropy, window={}, smoothing={}'.format(window, smoothing)
    ax.set_ylabel(f'Rolling entropy\n({title})')

    if save_to:
        plt.savefig(RESULT_FOLDER + os.sep + save_to)
    plt.show()

    return smoothed_entropies


def hist_3d(dfs, save_to=None):
    fig = plt.figure(figsize=(14, 7))
    plot_idx = 1
    for device, direction, df in utils.iterate_2layer_dict(dfs):
        ax = fig.add_subplot(1, 2, plot_idx, projection='3d')
        # Make data.
        hist, xedges, yedges = np.histogram2d(df['IAT'], df['pktLen'], bins=40, normed=False)

        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        # Construct arrays with the dimensions for the bars.
        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
        ax.set_axis_on()
        ax.set_title('{} | direction: {}'.format(device, direction))
        ax.set_xlabel('IAT, s')
        ax.set_ylabel('PS, bytes')
        plot_idx += 1
    plt.tight_layout()
    if save_to:
        plt.savefig(RESULT_FOLDER + os.sep + save_to)
    plt.show()


def hist_joint_dfs(dfs, save_to=None):
    fig = plt.figure(figsize=(13, 6))
    layout = (4, 8)
    # fig.subplots_adjust(wspace=0.5)

    fig_shifter = 0
    for device, direction, df in utils.iterate_2layer_dict_copy(dfs):

        # df = df.copy()
        df.index = [i for i in range(df.shape[0])]
        df.columns = ['IAT, s', 'PS, bytes']
        x = pd.Series(df['IAT, s'], name='')
        y = pd.Series(df['PS, bytes'], name='')

        kde_x = plt.subplot2grid(layout, (0, fig_shifter), colspan=3)
        kde_y = plt.subplot2grid(layout, (1, fig_shifter + 3), rowspan=3)
        scatter = plt.subplot2grid(layout, (1, fig_shifter), rowspan=3, colspan=3)

        sc_ax = sns.scatterplot(x='IAT, s', y='PS, bytes', data=df, ax=scatter, alpha=0.5)
        if fig_shifter == 4:
            sc_ax.set_ylabel('')
        kde_x.set_xlim(sc_ax.get_xlim())
        kde_y.set_ylim(sc_ax.get_ylim())
        kde_x.axis('off')
        kde_y.axis('off')
        kde_x.set_title('{} {}'.format(direction, device))

        sns.distplot(x, vertical=0, ax=kde_x)
        sns.distplot(y, vertical=1, ax=kde_y)
        fig_shifter += 4

    fig.tight_layout()
    if save_to:
        plt.savefig(RESULT_FOLDER + os.sep + save_to)


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
        utils.unpack_2layer_traffic_dict(plot_states)(states)
    if 't' in options:
        utils.unpack_2layer_traffic_dict(ts_analysis_plot)(states, lags=100)
    entropy = utils.unpack_2layer_traffic_dict(calc_windowed_entropy_discrete)(states)
    print('Entropy stats:\n{}\n'.format(pd.DataFrame(*list(entropy.values())).describe().loc[:, :]))
    if 'e' in options:
        utils.unpack_2layer_traffic_dict(plot_series_and_ma)(entropy, ma_window=6)
    if orig_states:
        print('KL divergence:\n{}'.format(
            pd.DataFrame(utils.unpack_2layer_traffic_dict(stat_metrics.get_KL_divergence_pmf)(orig_states, states)).T))


def ts_acfs_dfs(dfs, lags=200, save_to=None, style='bmh'):
    with plt.style.context(style):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
        plot_counter = 0
        for device, direction, y in utils.iterate_2layer_dict(dfs):
            smt.graphics.plot_acf(y, lags=lags, ax=axes[plot_counter, 0], title='ACF | {}'.format(direction))
            smt.graphics.plot_pacf(y, lags=lags, ax=axes[plot_counter, 1], title='Partial ACF | {}'.format(direction))
            plot_counter += 1
        plt.tight_layout()
        if save_to:
            plt.savefig(RESULT_FOLDER + os.sep + save_to)
        plt.show()


def _find_max_parameters(traffic_dfs):
    max_params = {'pktLen': 0, 'IAT': 0}

    for device in traffic_dfs:
        for parameter in ['pktLen', 'IAT']:
            max_params[parameter] = max(
                [max(traffic_dfs[device]['from'][parameter]), max(traffic_dfs[device]['to'][parameter])])

    return max_params
