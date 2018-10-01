
import pdb
import cProfile

import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import socket
import pickle
import dpkt
from dpkt.compat import compat_ord
import re
import scipy
import sklearn.neighbors
import sklearn

FIG_PARAMS = {'low_iat': 0.0000001,
              'high_iat': 200,
              'bins': 100}

def profile(func):
    """Decorator for run function profile"""
    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result
    return wrapper

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


def find_max_iat(device_traffic):

    # find the max IAT among all directions
    maxParam = {}
    for device in device_traffic:
        maxParam[device] = 0
        for direction in device_traffic[device]:
            for parameter in device_traffic[device][direction]:
                if (parameter == 'IAT') and device_traffic[device][direction][parameter]:
                    max_value = max(device_traffic[device][direction][parameter])
                    if max_value > maxParam[device]:
                        maxParam[device] = max_value
                else:
                    continue
    return maxParam


def define_figure_properties_per_device(device_traffic, logScale=True):
    global FIG_PARAMS

    
    device_props = {'IAT': {}, 'pktLen': {}}
    for direction in device_traffic:
        for parameter in device_traffic[direction]:

            try:
                max_from = max(device_traffic['from'][parameter])
            except ValueError:
                max_from = 0
            try:
                max_to = max(device_traffic['to'][parameter])
            except ValueError:
                max_to = 0

            maxParam = FIG_PARAMS['high_iat'] if not device_traffic[direction][parameter] else \
                max(max_from, max_to)

            if (parameter == 'pktLen'):
                device_props[parameter]['range'] = (0, 1500)
                device_props[parameter]['x_subscr'] = 'Payload Length, bytes'
                device_props[parameter]['bins'] = np.linspace(
                    0, 1500, FIG_PARAMS['bins'])
                device_props[parameter]['plot_numb'] = 0

            elif (parameter == 'IAT'):
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


def getXAxisValues(parameter, logScale, max_iat=0, traffic=[]):
    global FIG_PARAMS

    if max_iat > 0:
        upper_bound = max_iat
    elif not traffic:
        upper_bound = FIG_PARAMS['high_iat']
    else:
        upper_bound = max(traffic)

    if (parameter == 'pktLen'):
        x = np.linspace(1, 1500, 10000)

    elif (parameter == 'IAT') and not logScale:
        x = np.linspace(0, upper_bound, 10000)

    elif (parameter == 'IAT') and logScale:
        x = np.logspace(np.log10(FIG_PARAMS['low_iat']), upper_bound, 10000)

    return x


def gauss_function(x, x0, sigma, amp=1):
    '''
    gauss_function() defines normalized gaussian function
    '''
    # return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))
    return amp * np.exp(-0.5 * ((x-x0)/sigma)**2) * 1/(sigma * np.sqrt(2*np.pi))


def mod_addr(mac):
    '''
    replaces : and . with _ for addresses to enable saving to a disk
    '''
    return mac.replace(':', '_').replace('.', '_')


def ip_to_string(inet):
    """Convert inet object to a string
        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def is_mac_addr(string):
    if re.match("[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", string.lower()):
        return True
    else:
        return False


def is_ip_addr(string):
    try:
        socket.inet_aton(string)
        return True
    except socket.error:
        return False


def is_ip_port(string):
    if re.match("6553[0-6]|655[0-2][0-9]|65[0-4][0-9]{2}|6[0-4][0-9]{3}|[1-5][0-9]{4}|0?[1-9][0-9]{3}|0?0?[1-9][0-9]{2}|0?0?0?[1-9][0-9]|0{0,4}[1-9]", string):
        return True
    else:
        return False


def get_5_tuple_fields(string):

    try:
        tupleDict = {'proto': string.split(' ')[0],
                     'ip_s': string.split(' ')[1].split(':')[0],
                     'port_s': string.split(' ')[1].split(':')[1],
                     'ip_d': string.split(' ')[2].split(':')[0],
                     'port_d': string.split(' ')[2].split(':')[1]}

        return tupleDict

    except IndexError:
        print('Catched either empty or incorrect lines. Ignoring.')


def is_5_tuple(string):

    tup = get_5_tuple_fields(string)

    if re.match("udp|tcp", tup['proto'].lower()) and is_ip_port(tup['port_s']) and is_ip_port(tup['port_d']) and is_ip_addr(tup['ip_s']) and is_ip_addr(tup['ip_d']):
        return True
    else:
        return False


def mac_addr(address):
    """Convert a MAC address to a readable/printable string

       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return ':'.join('%02x' % compat_ord(b) for b in address)


def construct_new_dict(ref_dict):
    new_dict = {}
    for device in ref_dict:
        new_dict[device] = defaultdict(dict)
        for direction in ref_dict[device]:
            for parameter in ref_dict[device][direction]:
                new_dict[device][direction][parameter] = None

    return new_dict

def construct_new_dict_no_ts(ref_dict):
    new_dict = {}
    for device in ref_dict:
        new_dict[device] = defaultdict(dict)
        for direction in ref_dict[device]:
            for parameter in ref_dict[device][direction]:
                if parameter=='ts':
                    continue
                new_dict[device][direction][parameter] = None

    return new_dict

def get_IAT(TS):
    '''
    get_IAT() returns list with 'IAT', taking 'ts' list as the input
    '''
    iteration = 0
    IAT = []
    for ts in TS:
        if iteration == 0:
            IAT.append(0)
            tempIAT = ts
            iteration = iteration + 1
        else:
            IAT.append(ts - tempIAT)
            tempIAT = ts
    return IAT


def save_obj(obj, name):
    '''
    save_obj() saves python object to the file inside 'obj' directory
    '''
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    '''
    load_obj() loads python object from the file inside 'obj' directory
    '''
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)


def getAddressList(file, typeIdent):
    '''
    getAddressList() returns a list with identifiers to process, empty if 'all'
    specified
    '''

    addressList = []
    if file != 'all' or typeIdent != 'flow':
        with open(file, 'r') as f:
            print('Reading identifiers from the file...')
            for line in f:
                if line[0] == '#':
                    continue
                #print(line, end='')
                addressList.append(line.rstrip())
    return addressList


def get_data_below_percentile(device_traffic, percentile):

    new_traffic = []
    for entry in device_traffic:
        if entry > np.percentile(device_traffic, percentile):
            continue
        new_traffic.append(entry)

    return new_traffic


def get_data_within_percentiles(device_traffic, percentiles):

    new_traffic = []
    upper_bound = np.percentile(device_traffic, percentiles[1])
    lower_bound = np.percentile(device_traffic, percentiles[0])

    for entry in device_traffic:
        if entry > upper_bound or entry < lower_bound:
            continue
        new_traffic.append(entry)

    return new_traffic


def plot_hist(traffic, logScale=True, saveToFile=False):
    '''
    plot_hist() plots histograms from the layered dict with packet features.
    'logScale' affects only IAT parameter.
    'saveToFile' suppresses output and saves plots to the disk,
    'percentile' defines the percentile above which the data should be omitted
    (to exclude outliers)
    '''

    print('Preparing histograms for the traffic...')
    for device in traffic:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[12, 5])
        fig.suptitle('Histogram for '+device, fontsize=16)
        ax = ax.flatten()
        dev_fig_props = define_figure_properties_per_device(
            traffic[device], logScale)

        for direction in traffic[device]:
            # create a plain figure
            for parameter in traffic[device][direction]:

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
                        mod_addr(device)+'_'+'.svg')
            # else:
    plt.draw()
    plt.pause(0.001)
    input("Press any key to continue.")


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
            for parameter in traffic[device][direction]:

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
            print('Saved figure as {}'.format('hist_'+mod_addr(device).replace(' ','_')+'_'+'.svg'))
            plt.savefig('stat_figures'+os.sep+'hist_' +mod_addr(device)+'_'+'.svg')
            # else:
    plt.draw()
    plt.pause(0.001)
    input("Press any key to continue.")


def get_EM_values_dict(estimatedParameters, x_values):

    generated_em_values = construct_new_dict(estimatedParameters)

    for device in estimatedParameters:
        for direction in estimatedParameters[device]:
            for parameter in estimatedParameters[device][direction]:
                if not isinstance(estimatedParameters[device][direction][parameter],\
                    sklearn.mixture.gaussian_mixture.GaussianMixture):
                    continue
                    # pdb.set_trace()
                generated_em_values[device][direction][parameter] = \
                    getGaussianMixtureValues(estimatedParameters[device][direction][parameter],
                                             x_values[device][direction][parameter])

    return generated_em_values


def getGaussianMixtureValues(gmm, gmm_x):
    # Construct function manually as sum of gaussians
    gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
    # pdb.set_trace()
    for w, m, var in zip(gmm.weights_.ravel(), gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())):
        gauss = gauss_function(x=gmm_x, x0=m, sigma=var)
        #print(np.trapz(gauss, gmm_x))
        normalizer = np.trapz(gauss, gmm_x)
        if normalizer > 0.000001:
            gmm_y_sum += gauss * w / normalizer
        else:
            gmm_y_sum += gauss * w

    return gmm_y_sum


def printGaussianMixtureParameters(gmm, componentWeightThreshold):
    for w, m, var in zip(gmm.weights_.ravel(), gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())):
        if w < componentWeightThreshold:
            continue
        print('W: {}, M: {}, V: {}'.format(w, m, var))


def printEstimatedParametersEM(estimationsEM, componentWeightThreshold=0.02):
    '''
    printEstimatedParametersEM() prints mixture components to STDOUT
    '''
    for device in estimationsEM:
        for direction in estimationsEM[device]:
            print('_______________________\n{1} {0}'.format(device, direction))
            for parameter in estimationsEM[device][direction]:
                if not isinstance(estimationsEM[device][direction][parameter],\
                                  sklearn.mixture.gaussian_mixture.GaussianMixture):
                    continue
                print("Parameter %s:" % parameter)
                gmm = estimationsEM[device][direction][parameter]
                for w, m, var in zip(gmm.weights_.ravel(), gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())):
                    if w < componentWeightThreshold:
                        continue
                    print(
                        'W: {0:1.3f}, M: {1:5.6f}, V: {2:5.8f}'.format(w, m, var))


def plot2D(traffic):
    '''
    plot2D() vizualizes 'IAT' and 'pktLen' parameters as a scatter plot
    '''
    for device in traffic:
        for direction in traffic[device]:
            print(direction+' '+device)
            fig = plt.figure()
            ax = plt.gca()
            ax.plot(traffic[device][direction]['IAT'], traffic[device][direction]
                    ['pktLen'], 'o', c='blue', alpha=0.05, markeredgecolor='none')
            ax.set_xscale('log')
            ax.set_xlabel('IAT, s')
            ax.set_ylabel('Packet size, bytes')
            ax.grid(True)
            ax.set_title(direction+' '+device)


def getTrafficExtremeValues(traffic):
    '''
    getTrafficExtremeValues() extracts extreme values i.e. max and 
    min values of 'IAT' and 'pktLen' from traffic dict and returns 
    dict with them
    '''

    extremeValues = defaultdict(dict)
    for device in traffic:
        for direction in traffic[device]:
            extremeValues[device][direction] = {'pktLen': {}, 'IAT': {}}
            for parameter in ['pktLen', 'IAT']:
                extremeValues[device][direction][parameter] = {
                    'max': 0, 'min': 0}

                try:
                    extremeValues[device][direction][parameter]['min'] = min(
                        traffic[device][direction][parameter])
                except ValueError:
                    extremeValues[device][direction][parameter]['min'] = 0
                try:
                    extremeValues[device][direction][parameter]['max'] = max(
                        traffic[device][direction][parameter])
                except ValueError:
                    extremeValues[device][direction][parameter]['max'] = 0

    return extremeValues


def get_pcap_filename(args):
    return args.p.split('/')[len(args.p.split('/'))-1].split('.')[0]


def print_parameters(header, dictWithDevices):
    print('\n')
    print(header)
    for device in dictWithDevices:
        print(device)
        for direction in dictWithDevices[device]:
            for parameter in dictWithDevices[device][direction]:
                par = dictWithDevices[device][direction][parameter]
                if isinstance(par, float):
                    formatPar = '{0:8s} {1:6s}: {2:3.3f}'
                else:
                    formatPar = '{0:8s} {1:6s}: {2}'
                print(formatPar.format(parameter, direction, par))
    # print('\n')


def get_KDE_estimators_scipy(traffic):
    '''
    get_KDE_scipy() returns dict with estimated kernel density function,
    where it was estimated on data with values less than 95 percentile,
    in order to exclude outliers during bandwidth estimation process within 
    the scipy function
    '''

    print('Performing kernel density estimations...')
    kde_estimators = {}
    for device in traffic:
        kde_estimators[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:

                try:
                    kde_estimators[device][direction][parameter] = scipy.stats.gaussian_kde(
                    traffic[device][direction][parameter], bw_method='scott')
                except ValueError:
                    print('Could not esimate KDE for {} {} {}'.format(direction,device,parameter))
                    kde_estimators[device][direction][parameter] = traffic[device][direction][parameter]
                    continue

    return kde_estimators


def get_KDE_values(kde_estimators, x_values):

    kde_values = construct_new_dict(kde_estimators)
    for device in kde_estimators:
        for direction in kde_estimators[device]:
            for parameter in kde_estimators[device][direction]:
                estimator = kde_estimators[device][direction][parameter]
                #print(type(estimator))
                if isinstance(estimator, sklearn.neighbors.kde.KernelDensity):
                    kde_values[device][direction][parameter] = np.exp(estimator.score_samples(
                        np.array(x_values[device][direction][parameter]).reshape(-1,1)))
                elif isinstance(estimator, scipy.stats.kde.gaussian_kde):

                    kde_values[device][direction][parameter] = estimator.evaluate(
                        x_values[device][direction][parameter])
                else:
                    print('KDE: omitting {} {} {} for plotting'.format(direction, device, parameter))

    return kde_values


def get_EM_values_sklearn(estimatedParameters, sample_number=1000):

    generated_em_values = construct_new_dict(estimatedParameters)

    for device in estimatedParameters:
        for direction in estimatedParameters[device]:
            for parameter in estimatedParameters[device][direction]:
                generated_em_values[device][direction][parameter] = estimatedParameters[device][direction][parameter].sample(
                    sample_number)

    return generated_em_values


def get_EM_values_manual(estimatedParameters, trafficExtremeValues=[], packetNumber=1000):
    '''
    get_EM_values() generates packet properties given the EM-estimations,
    the max and min values of the parameters and the number of packets to generate
    '''

    generated_em_values = construct_new_dict(estimatedParameters)

    for device in estimatedParameters:
        for direction in estimatedParameters[device]:
            for parameter in estimatedParameters[device][direction]:

                mix = estimatedParameters[device][direction][parameter]
                if trafficExtremeValues:
                    paramLimit = trafficExtremeValues[device][direction][parameter]
                # generate parameters for the number of packets from the original dump
                for _ in range(packetNumber):
                    # select the component number that we are drawing sample from (Multinoulli)
                    component = np.random.choice(
                        np.arange(0, mix.n_components), p=mix.weights_)

                    # generate parameter until it fits the limits
                    suitable = False
                    while not suitable:
                        if parameter == 'pktLen':
                            # packet length in bytes must be integer
                            genParam = round(np.asscalar(random.gauss(
                                mix.means_[component], np.sqrt(mix.covariances_[component]))))
                        else:
                            genParam = np.asscalar(random.gauss(
                                mix.means_[component], np.sqrt(mix.covariances_[component])))

                        if not trafficExtremeValues or ((genParam <= paramLimit['max']) and (genParam >= paramLimit['min'])):
                            suitable = True

                    generated_em_values[device][direction][parameter].append(
                        genParam)

                # replace the first value of IAT with 0 for consistency
                if parameter == 'IAT':
                    generated_em_values[device][direction][parameter][0] = 0

    return generated_em_values


def get_x_values_dict(traffic, logScale=False):

    x_values = construct_new_dict_no_ts(traffic)
    max_iat = find_max_iat(traffic)
    for device in traffic:
        for direction in traffic[device]:
            for parameter in traffic[device][direction]:
                if parameter == 'ts':
                    continue

                x_values[device][direction][parameter] = getXAxisValues(
                    parameter=parameter, traffic=traffic[device][direction][parameter], max_iat=max_iat[device], logScale=logScale)

    return x_values
