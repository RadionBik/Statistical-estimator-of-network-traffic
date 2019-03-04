
from traffic_helpers import *
from gmm_helpers import *

import matplotlib.pyplot as plt

FIG_PARAMS = {'low_iat': 0.0000001,
              'high_iat': 200,
              'high_pktlen' : 1500,
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
        pd.Series([df['pktLen'].autocorr(lag) for lag in range( round(len(df['pktLen'])/2)) ] ).plot(ax=axes[0,1], grid=True)
        axes[0,1].xaxis.set_label_text('Lag')
        df['pktLen'].plot.box( ax=axes[0,0], vert=False, grid=True)
        axes[0,0].xaxis.set_label_text('bytes')
        axes[0,1].yaxis.set_label_text('Autocorrelation')
        pd.Series([df['IAT'].autocorr(lag) for lag in range( round(len(df['IAT'])/2)) ] ).plot(ax=axes[1,1], grid=True)
        df['IAT'].plot.box( ax=axes[1,0], vert=False, grid=True)
        axes[1,0].xaxis.set_label_text('seconds')
        axes[1,1].yaxis.set_label_text('Autocorrelation')
        axes[1,1].xaxis.set_label_text('Lag')
        if saveToFile:
                plt.savefig('stat_figures'+os.sep+'stat_props_' +
                            mod_addr(device)+'_'+direction+'_'+saveToFile+'.pdf')

def print_stat_properties(traffic_dfs, model='HMM'):
    print('----------------\nModel {}:'.format(model))
    for device, direction, df in iterate_traffic_dict(traffic_dfs):
        print('{} {}:\n {}\n'.format(direction, device, df.describe().loc[['mean','std', '50%'],:]))

def get_goodput(traffic_dfs, resolution='1S', plot=True, saveToFile=None):
    goodput_dfs = construct_dict_2_layers(traffic_dfs)
    plt.figure()
    for device, direction, df in iterate_dfs_plus(traffic_dfs):
        #replace indexes with DateTime format
        df.index = pd.to_datetime(df.IAT.cumsum(), unit='s')
        #df.pop('ts')
        goodput_dfs[device][direction] = df.resample(resolution).sum()
        #plt.figure()
        if plot:
            ax = (goodput_dfs[device][direction]['pktLen']/1024).plot(grid=True, label=direction, lw=3)
            ax.set(xlabel='time',ylabel='KB per '+resolution, title=f'Goodput for {device}')
            ax.legend()

            if saveToFile:
                plt.savefig('stat_figures'+os.sep+'goodput_' +
                        mod_addr(device)+'_'+saveToFile+'.pdf')

        
    return goodput_dfs

def plot_2D_hist(traffic_dfs, parameters=['pktLen','IAT'], logScale=False, states=None, saveToFile=None):
    '''
    plot2D() vizualizes 'IAT' and 'pktLen' parameters as a scatter plot
    '''
    for device in traffic_dfs:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        for dir_numb, direction in enumerate(traffic_dfs[device]):
            df = traffic_dfs[device][direction]
            #fig = plt.figure()
            #ax[dir_numb] = plt.gca()
            if states:
                colors = states[device][direction]
            else:
                colors = 'b'
            ax[dir_numb].scatter(df[parameters[1]],  df[parameters[0]], c=colors,label=f"direction '{direction}'", alpha=0.2)#, markeredgecolor='none', )
            if logScale:
                ax[dir_numb].set_xscale('log')
            ax[dir_numb].set_xlabel('IAT, s')
            ax[dir_numb].set_ylabel('Payload size, bytes')
            ax[dir_numb].legend(loc='best')
            ax[dir_numb].grid(True)
            ax[dir_numb].set_title(device)

        if saveToFile:
                plt.savefig('stat_figures'+os.sep+'hist2d_' +
                            mod_addr(device)+'_'+saveToFile+'.pdf')




def plot_hist(traffic, logScale=True, saveToFile=None):
    '''
    plot_hist() plots histograms from the layered dict with packet features.
    'logScale' affects only IAT parameter.
    'saveToFile' suppresses output and saves plots to the disk,
    (to exclude outliers)
    '''

    print('Preparing histograms for the traffic...')
    for device in traffic:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.suptitle('Histogram for '+device, fontsize=16)
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
                ax[plotNumb].set_title('direction: {} ({} packets)'.format(direction,len(deviceData)))
                ax[plotNumb].grid(True)
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)
        if saveToFile:
            plt.savefig('stat_figures'+os.sep+'hist_' +
                        mod_addr(device)+'_'+saveToFile+'.pdf')
            # else:
    plt.draw()
    #plt.pause(0.001)
    #input("Press any key to continue.")

def plot_hist_kde_em(traffic, kde_estimators, em_estimators=None, logScale=True, saveToFile=False, min_samples_to_estimate=15):
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

    #em_values = get_EM_values(em_estimators)
    em_values = get_EM_values_dict(em_estimators, x_values)


    for device in traffic:
        
        if len(traffic[device]['from']['ts'])<min_samples_to_estimate or len(traffic[device]['to']['ts'])<min_samples_to_estimate:
            row_numb = 1
        else:
            row_numb = 2

        fig, ax = plt.subplots(nrows=row_numb, ncols=2, figsize=[12, 5])
        fig.suptitle('Estimations for '+device, fontsize=16)
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
                    if row_numb==1:
                        plotNumb = plotNumb - 2

                x = x_values[device][direction][parameter][1:-1]
                
                # plot histogram
                ax[plotNumb].hist(deviceData, bins=dev_fig_props[parameter]['bins'],
                                  range=dev_fig_props[parameter]['range'], density=True)

                if kde_values[device][direction][parameter] is not None:

                    ax[plotNumb].plot(x, kde_values[device][direction][parameter][1:-1], color="red", lw=1.5, label='KDE', linestyle='--')

                if em_values[device][direction][parameter] is not None:
                    ax[plotNumb].plot(x, em_values[device][direction][parameter][1:-1], color="black", lw=1.5, label='EM', linestyle='--')

                ax[plotNumb].legend()

                # Annotate diagram
                ax[plotNumb].set_xlabel(dev_fig_props[parameter]['x_subscr'])

                if (parameter == 'IAT') and logScale:
                    ax[plotNumb].set_xscale("log")
                ax[plotNumb].set_title('direction: {} ({} packets)'.format(direction, len(deviceData)))
                ax[plotNumb].grid(True)
                #ax[plotNumb].set_ylim([0, max(max(kde_values[device][direction][parameter]),max(em_values[device][direction][parameter]))])
                fig.tight_layout()
                fig.subplots_adjust(top=0.88)
        if saveToFile:
            figure_name = 'stat_figures'+os.sep+'hist_'+mod_addr(device).replace(' ','_')+'_'+'.svg'
            print('Saved figure as {}'.format(figure_name))
            plt.savefig(figure_name)
            # else:
    plt.draw()
    plt.pause(0.001)
    input("Press any key to continue.")

def getXAxisValues(parameter, logScale=False, max_param=None, traffic=None):

    if max_param:
        upper_bound = max_param
    elif isinstance(traffic, pd.Series):
        upper_bound = max(traffic)
    else:
        upper_bound = 200

    if (parameter == 'pktLen'):
        x = np.linspace(1, upper_bound, 1000)

    elif (parameter == 'IAT') and not logScale:
        x = np.linspace(0, upper_bound, 1000)

    elif (parameter == 'IAT') and logScale:
        x = np.logspace(np.log10(0.000001), upper_bound, 1000)

    return x

def get_x_values_dict(traffic, logScale=False):

    x_values = construct_new_dict_no_ts(traffic)
    max_params = find_max_parameters(traffic)
    for device in traffic:
        for direction in traffic[device]:
            for parameter in ['pktLen', 'IAT']:
                #if parameter == 'ts':
                #    continue

                x_values[device][direction][parameter] = getXAxisValues(
                    parameter=parameter, traffic=traffic[device][direction][parameter], max_param=max_params[parameter], logScale=logScale)

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
                    max(max_value[parameter].values())+50
                except ValueError:
                    maxParam = FIG_PARAMS['high_pktlen'] if device_traffic[direction][parameter].empty else \
                    max(max_value[parameter].values())+50

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
    #hist_prop = {}
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