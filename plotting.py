#from traffic_helpers import *
from gmm_helpers import *
from ts_helpers import *

import seaborn as sns
import matplotlib.pyplot as plt
import os

FIG_PARAMS = {'low_iat': 0.0000001,
              'high_iat': 200,
              'high_pktlen': 1500,
              'bins': 100}


def plot_stat_properties(traffic_dfs, saveToFile=None):
    '''
    draws statistical figures for a traffic DataFrame, includes BoxPlot and Autocorrelation
    for PktLen and IAT
    '''

    for device, direction, df in iterate_traffic_dict(traffic_dfs):
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
        if saveToFile:
            plt.savefig('stat_figures' + os.sep + 'stat_props_' +
                        mod_addr(device) + '_' + direction + '_' + saveToFile + '.pdf')


def print_stat_properties(traffic_dfs, model='HMM'):
    print('----------------\nModel {}:'.format(model))
    for device, direction, df in iterate_traffic_dict(traffic_dfs):
        print('{} {}:\n {}\n'.format(direction, device, df.describe().loc[['mean', 'std', '50%'], :]))


def hist_2d_dfs(traffic_dfs, parameters=('pktLen', 'IAT'), logScale=False, states=None, saveToFile=None):
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
            if logScale:
                ax[dir_numb].set_xscale('log')
            ax[dir_numb].set_xlabel('IAT, s')
            ax[dir_numb].set_ylabel('Payload size, bytes')
            ax[dir_numb].legend(loc='best')
            ax[dir_numb].grid(True)
            ax[dir_numb].set_title(device)

        if saveToFile:
            plt.savefig('stat_figures' + os.sep + 'hist2d_' +
                        mod_addr(device) + '_' + saveToFile + '.pdf')


def hist_dfs(traffic, logScale=True, saveToFile=None):
    '''
    hist_dfs() plots histograms from the layered dict with packet features.
    'logScale' affects only IAT parameter.
    'saveToFile' suppresses output and saves plots to the disk,
    (to exclude outliers)
    '''
    with plt.style.context('bmh'):
        print('Preparing histograms for the traffic...')
        for device in traffic:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
            fig.suptitle('Histogram for ' + device, fontsize=16)
            ax = ax.flatten()
            dev_fig_props = define_figure_properties_per_device(
                traffic[device], logScale)

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

                    if (parameter == 'IAT') and logScale:
                        ax[plotNumb].set_xscale("log")
                    ax[plotNumb].set_title('direction: {} ({} packets)'.format(direction, len(deviceData)))
                    ax[plotNumb].grid(True)
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.88)
            if saveToFile:
                plt.savefig('stat_figures' + os.sep + 'hist_' +
                            mod_addr(device) + '_' + saveToFile + '.pdf')
                # else:
        plt.show()
    # plt.pause(0.001)
    # input("Press any key to continue.")


def plot_hist_kde_em(traffic, kde_estimators, em_estimators=None, logScale=True, saveToFile=False,
                     min_samples_to_estimate=15):
    '''
    plot_hist_kde_em() plots histograms, KDEs and EM estimations from the 
    layered dict with packet features.
    'logScale' affects only IAT parameter.
    'saveToFile' suppresses output and saves plots to the disk,
    'percentile' defines the percentile above which the data should be omitted
    (to exclude outliers)
    '''

    print('Preparing plots for the estimations...')
    x_values = get_x_values_dict(traffic, logScale)

    kde_values = get_KDE_values(kde_estimators, x_values)

    # em_values = get_EM_values(em_estimators)
    em_values = get_EM_values_dict(em_estimators, x_values)

    for device in traffic:

        if len(traffic[device]['from']['ts']) < min_samples_to_estimate or len(
                traffic[device]['to']['ts']) < min_samples_to_estimate:
            row_numb = 1
        else:
            row_numb = 2

        fig, ax = plt.subplots(nrows=row_numb, ncols=2, figsize=[12, 5])
        fig.suptitle('Estimations for ' + device, fontsize=16)
        ax = ax.flatten()
        dev_fig_props = define_figure_properties_per_device(
            traffic[device], logScale)

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

                if (parameter == 'IAT') and logScale:
                    ax[plotNumb].set_xscale("log")
                ax[plotNumb].set_title('direction: {} ({} packets)'.format(direction, len(deviceData)))
                ax[plotNumb].grid(True)
                # ax[plotNumb].set_ylim([0, max(max(kde_values[device][direction][parameter]),max(em_values[device][direction][parameter]))])
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)
        if saveToFile:
            figure_name = 'stat_figures' + os.sep + 'hist_' + mod_addr(device).replace(' ', '_') + '_' + '.svg'
            print('Saved figure as {}'.format(figure_name))
            plt.savefig(figure_name)
            # else:
    plt.draw()
    plt.pause(0.001)
    input("Press any key to continue.")


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
    x_values = construct_new_dict_no_ts(traffic)
    max_params = find_max_parameters(traffic)
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
    global FIG_PARAMS

    device_props = {'IAT': {}, 'pktLen': {}}
    max_value = defaultdict(dict)
    for direction in device_traffic:
        for parameter in ['pktLen', 'IAT']:

            try:
                max_value[parameter][direction] = max(device_traffic[direction][parameter])
            except ValueError:
                max_value[parameter][direction] = 0

            if (parameter == 'pktLen'):

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

            elif (parameter == 'IAT'):

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
    global FIG_PARAMS
    fig_prop = {}
    # hist_prop = {}
    if (parameter == 'pktLen'):
        fig_prop['param'] = 'Payload Length'
        fig_prop['unit'] = ", bytes"
        fig_prop['range'] = (0, 1500)
        fig_prop['bins'] = np.linspace(0, 1500, FIG_PARAMS['bins'])
        fig_prop['plot_numb'] = 0
    elif (parameter == 'IAT'):
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


def plot_iat_pktlen(df):
    df = df.copy()
    # plt.figure()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    df.index = [i for i in range(df.shape[0])]
    df['IAT'].plot(ax=axes[0])
    axes[0].xaxis.set_label_text('packet #')
    axes[0].yaxis.set_label_text('IAT, s')
    df['pktLen'].plot(ax=axes[1])
    axes[1].xaxis.set_label_text('packet #')
    axes[1].yaxis.set_label_text('Packet Size, bytes')
    plt.show()


def plot_acf_df(df, lags=None):
    with plt.style.context('bmh'):
        if not lags:
            lags = round(len(df['pktLen']))
        fig, axes = plt.subplots(nrows=1, ncols=2)
        smt.graphics.plot_acf(df['IAT'], lags=lags, ax=axes[0])
        # pd.Series([df['IAT'].autocorr(lag)
        #           for lag in range(lags)]).plot(ax=axes[0], grid=1)
        axes[0].xaxis.set_label_text('Lag')
        axes[0].yaxis.set_label_text('ACF, IAT')
        smt.graphics.plot_acf(df['pktLen'], lags=lags, ax=axes[1])
        # pd.Series([df['pktLen'].autocorr(lag)
        #           for lag in range(lags)]).plot(ax=axes[1], grid=1)
        axes[1].xaxis.set_label_text('Lag')
        axes[1].yaxis.set_label_text('ACF, Packet Size')
        plt.show()


def plot_dfs_acf(dfs, lags=500, saveToFile=None):
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=[14, 7])
    plot_counter = 0
    for device, direction, df in iterate_dfs_plus(dfs):
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
    if saveToFile:
        plt.savefig('stat_figures' + os.sep + saveToFile)

    plt.show()


def goodput(df, resolution='1S', plot=True, saveToFile=None):
    plt.figure()
    # replace indexes with DateTime format
    df.index = pd.to_datetime(df.IAT.cumsum(), unit='s')
    goodput_df = df.resample(resolution).sum()

    ax = (goodput_df['pktLen'] / 1024).plot(grid=True, lw=3)  # label=direction
    # for {device}')
    ax.set(xlabel='time', ylabel='KB per ' + resolution, title=f'Goodput')
    # ax.legend()
    plt.show()
    if saveToFile:
        plt.savefig('stat_figures' + os.sep + saveToFile)


def goodput_dfs(dfs, resolution='1S', saveToFile=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[12, 5])
    plot_counter = 0
    for device, direction, df in iterate_dfs_plus(dfs):
        # replace indexes with DateTime format
        df.index = pd.to_datetime(df.IAT.cumsum(), unit='s')
        goodput_df = df.resample(resolution).sum()

        # for {device}')
        (goodput_df['pktLen'] / 1024).plot(grid=True, lw=3, ax=ax[plot_counter])  # label=direction
        ax[plot_counter].set(xlabel='time',
                             ylabel='Goodput kB/' + resolution,
                             title=direction + ' ' + device)
        # ax.legend()
        plot_counter += 1
    if saveToFile:
        plt.savefig('stat_figures' + os.sep + saveToFile)

    plt.show()


def entropies_dfs(traffic_dfs,
                  bar=True,
                  smoothing=10,
                  kde_bins=500,
                  window=50,
                  saveToFile=None):
    entrs = []
    legends = []
    params = []
    plt.figure()
    for device, direction, parameter, ser in iterate_traffic_3_layers(traffic_dfs):
        entropy = calc_windowed_entropy_cont(traffic_dfs[device][direction][parameter], kde_bins=kde_bins,
                                             window=window)
        mean_entropy = np.mean(entropy)
        entrs.append(mean_entropy)
        if parameter == 'pktLen':
            parameter = 'PS'
        params.append(direction + ', ' + parameter)
        legends.append('{:4} | {:3} | avg={:1.2f}'.format(direction, parameter, mean_entropy))
        ax = pd.Series(entropy).rolling(smoothing, center=True).mean().plot(grid=True)
        # plot_series_and_ma(entropy, ma_window=5)

    ax.legend(legends)
    ax.set_title('{} | Rolling entropy, window={}, smoothing={}'.format(device, window,
                                                                        smoothing))
    entr_series = pd.Series(entrs, index=params)
    # print(entr_series)
    if bar:
        plt.figure()
        ax = entr_series.plot(kind='bar', grid=True, rot=30)
        ax.set_title('Average entropy of parameters')
        # ax.set_xticks(legends)
    if saveToFile:
        plt.savefig('stat_figures' + os.sep + saveToFile)
    plt.show()

    return entr_series


def hist_3d(dfs, save_to=None):
    from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib import cm
    # from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = plt.figure(figsize=(14, 7))
    plot_idx = 1
    for device, direction, df in iterate_traffic_dict(dfs):
        ax = fig.add_subplot(1, 2, plot_idx, projection='3d')
        # Make data.
        hist, xedges, yedges = np.histogram2d(df['IAT'], df['pktLen'], bins=40, normed=0)

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
        plt.savefig('stat_figures' + os.sep + save_to)
    plt.show()


def hist_joint_dfs(dfs, save_to=None):
    fig = plt.figure(figsize=(13, 6))
    layout = (4, 8)
    # fig.subplots_adjust(wspace=0.5)

    fig_shifter = 0
    for device, direction, df in iterate_dfs_plus(dfs):

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
        plt.savefig('stat_figures' + os.sep + save_to)
