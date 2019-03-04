#! /usr/bin/python3

'''
usage example:

./stat_estimator.py -p ../voiceTraces/skypeLANhome.pcap -i flow --plot

./stat_estimator.py -p ../voiceTraces/skypeLANhome.pcap -i flow -f all -n auto

'''

import argparse
import os
from sklearn import mixture
import numpy as np
import random
from collections import defaultdict, Counter
from traffic_helpers import * 
import dpkt
from subprocess import Popen, PIPE, run
import re
import socket
from sklearn.neighbors import KernelDensity
#from sklearn.model_selection import GridSearchCV

def modifyTrafficDict(traffic):
    '''
    temporal usage as an interface between this and the old program
    '''
    traffic_per_device = defaultdict(dict)
    for direction in traffic:
        for group in traffic[direction]:
            for device in traffic[direction][group]:
                traffic_per_device[device][direction] = traffic[direction][group][device]
    return traffic_per_device

def get_voip_flow_list(pcapfile):
    
    pipe = Popen(["./ndpi_arch", "-i", pcapfile, "-v2"], stdout=PIPE)
    raw = pipe.communicate()[0].decode("utf-8")

    reg = re.compile(
        r'(UDP) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) <?->? (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) \[proto: [\d+\.]*\d+\/(RTP|Skype.SkypeCall)*\]')

    flows = []
  
    for transp_proto, ip1, port1, ip2, port2, app_proto in re.findall(reg, raw):
        flows.append(transp_proto+' '+ip1+':'+port1+' '+ip2+':'+port2)

    if flows:
        print('Detected the following VoIP flows:')
        for flow in flows:
            print(flow)
    else:
        exit('No VoIP flows detected')

    return flows

def getFlowList(pcapfile):
    print('Extracting flow identifiers from {}...'.format(pcapfile))
    keys = []
    counter= 0
    for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
        eth = dpkt.ethernet.Ethernet(raw)
        ip = eth.data
        #check if the packet is IP, TCP, UDP
        if not isinstance(ip, dpkt.ip.IP):
            continue
        seg = ip.data
        if isinstance(seg, dpkt.tcp.TCP):
            transp_proto = "TCP"
        elif isinstance(seg, dpkt.udp.UDP):
            transp_proto = "UDP"
        else:
            continue

        new_key = (transp_proto, (((ip.src, seg.sport),(ip.dst, seg.dport))))
        if new_key not in keys:
            #print(new_key)
            keys.append(new_key)
    
    #define set of unique keys
    keys_set = set()
    for key in keys:
        key_set = (key[0],  frozenset( (key[1][0], key[1][1]) )  )
        #print(key_set)
        keys_set.add( key_set )
    
    #remove duplicate keys, preserving the order and using the keys_set,
    #workaround to avoid uncertain order of using set-type keys.
    #not the best solution 
    identifiers=[]
    for orig_key in keys_set:
        found_key = False
        for key in keys:
            if not found_key:
                if orig_key==(key[0],  frozenset( (key[1][0], key[1][1]) )  ):
                    identifiers.append(key[0]+' '+ip_to_string(list(key[1])[0][0])+':'+str(list(key[1])[0][1])+' '+ip_to_string(list(key[1])[1][0])+':'+str(list(key[1])[1][1]))
                    found_key = True

    return identifiers

def extract_flow_stats(pcapfile, flows, min_samples_to_estimate=0, payloadOnly=True):
    
    print('Extracting flow features from {}...'.format(pcapfile))
    #create the layered dict
    traffic = {}
    if not flows:
        flows = getFlowList(pcapfile)

    
    identifiersType = checkIdentifiersType(flows)
    if type(flows) is str:
        flows = [flows]
    for identifier in flows:
        if identifiersType[identifier]!='flow':
                continue
        #traffic[identifier] = {'from': {'ts': [],'pktLen': [],'IAT': []}, 
        #                       'to': {'ts': [],'pktLen': [],'IAT': []}}
        traffic[identifier] = {'from': {'ts': [],'pktLen': []}, 
                               'to': {'ts': [],'pktLen': []}}


    for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
        eth = dpkt.ethernet.Ethernet(raw)
        ip = eth.data

        #skip ARP, ICMP, etc.
        if not isinstance(ip, dpkt.ip.IP):
            continue
        if not isinstance(ip.data, dpkt.tcp.TCP) and not isinstance(ip.data, dpkt.udp.UDP):
            continue
        #filter out segments and datagrams without payload (e.g. SYN, SYN/ACK, etc.)
        if payloadOnly and len(ip.data.data) == 0:
            continue

        seg = ip.data
        key_packet_from = (isinstance(seg, dpkt.tcp.TCP),ip_to_string(ip.src),seg.sport,ip_to_string(ip.dst),seg.dport)
        key_packet_to = (isinstance(seg, dpkt.tcp.TCP),ip_to_string(ip.dst),seg.dport, ip_to_string(ip.src),seg.sport)
        for identifier in traffic:
            tup = get_5_tuple_fields(identifier)
            key_ident = (tup['proto']=='TCP',tup['ip_s'],int(tup['port_s']),tup['ip_d'],int(tup['port_d']))
            if key_ident==key_packet_from:
                direction = 'from'

            elif key_ident==key_packet_to:
                direction = 'to'
            else:
                continue
                   
            traffic[identifier][direction]['ts'].append(ts)
            if payloadOnly:
                traffic[identifier][direction]['pktLen'].append(len(seg.data))
                break
            else:
                traffic[identifier][direction]['pktLen'].append(len(ip))
                break

    
    #print the number of packets and remove empty identifiers
    emptyIdentifiers = set()
    for identifier in traffic:
        if len(traffic[identifier]['from']['ts']) < min_samples_to_estimate and \
            len(traffic[identifier]['to']['ts']) < min_samples_to_estimate:
                emptyIdentifiers.add(identifier)
    
    for identifier in emptyIdentifiers:
        traffic.pop(identifier, None)
        
    if not traffic:
        exit('Could not find flows with # of packets > {}!'.format(min_samples_to_estimate))

    print('Found the following flows with # of packets > {}:'.format(min_samples_to_estimate))
    for identifier in traffic:
        for direction in traffic[identifier]:
            #traffic[identifier][direction]['IAT'] = get_IAT(traffic[identifier][direction]['ts'])
            print('{} pkt number {}: {}'.format(identifier,direction,len(traffic[identifier][direction]['ts'])))

    return traffic

def checkIdentifiersType(identifiers):

    identifiersType = {}
    for host in identifiers:    
        if is_mac_addr(host):
            identifiersType[host] = 'MAC'
        elif is_ip_addr(host):
            identifiersType[host] = 'IP'
        elif is_5_tuple(host):
            identifiersType[host] = 'flow'
        else:
            print('Identifier {} is neither MAC or IP or a flow. Ignoring'.format(host))
            
    #print(identifiersType)
    return identifiersType

def extractHostStatsDpkt(pcapfile, hosts=[], setIdentifier='IP', payloadOnly=True, min_samples_to_estimate=15):
    '''
    extractHostsFromPcapDpkt() uses the dpkt lib to extract packet features of the 
    desired hosts.
    'macAddress' searches hosts by MAC addresses if True, or by IP if False
    
    TODO
    1. add IP identifier capability
    2. add tun int capability
    
    '''
   
    #create the layered dict
    traffic = {}
    if type(hosts) is str:
        hosts = [hosts]
    for identifier in hosts:
            traffic[identifier] = {'from': {'ts': [],'pktLen': [],'IAT': []}, 
                                     'to': {'ts': [],'pktLen': [],'IAT': []}}
    

    identifiersType = checkIdentifiersType(hosts)

    #TODO ineffective implementation, make it to iterate once to fill the traffic
    for host in hosts:
        
        identifierType = identifiersType[host]

        if identifierType != setIdentifier:
            continue
   
        for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
            eth = dpkt.ethernet.Ethernet(raw)
            ip = eth.data

            #skip ARP, ICMP, etc.
            if not isinstance(ip, dpkt.ip.IP):
                continue
            #filter out segments and datagrams without payload (e.g. SYN, SYN/ACK, etc.)
            if payloadOnly and len(ip.data.data) == 0:
                continue

            if identifierType == 'MAC':
                identifierFrom = mac_addr(eth.src)
                identifierTo = mac_addr(eth.dst)
            else:
                identifierFrom = ip_to_string(eth.data.src)
                identifierTo = ip_to_string(eth.data.dst)

            if identifierFrom == host:
                direction = 'from'
            elif identifierTo == host:
                direction = 'to'
            else:
                continue

            traffic[identifierTo]['to']['ts'].append(ts)
            if payloadOnly:
                traffic[identifierTo]['to']['pktLen'].append(len(ip.data.data))
            else:
                traffic[identifierTo]['to']['pktLen'].append(len(ip))

    emptyIdentifiers = set()
    for identifier in traffic:
        for direction in traffic[identifier]:
            if len(traffic[identifier][direction]['ts']) < min_samples_to_estimate:
                emptyIdentifiers.add(identifier)
    
    for identifier in emptyIdentifiers:
        traffic.pop(identifier, None)
        
    print('Found the following non-empty identifiers:')
    for identifier in traffic:
        for direction in traffic[identifier]:
            traffic[identifier][direction]['IAT'] = get_IAT(traffic[identifier][direction]['ts'])
            print('{} pkt number {}: {}'.format(identifier,direction,len(traffic[identifier][direction]['ts'])))

    return traffic

def getNotEmptyIdentifiers(identifier, traffic):
    notEmptyIdentifiers = []
    for identifier in traffic:
        for direction in traffic[identifier]:
            pass
        if len(traffic[identifier][direction]['pktLen']) > 0:
            notEmptyIdentifiers.append(identifier)

    return notEmptyIdentifiers
 
def estimateParametersEM(traffic, componentNumb=5):
    '''
    estimateParametersEM() estimates statistical properties (Gauusian Mixtures) 
    via EM-algorithm for IAT and pktLen parameters for each device/flow and returns
    dict with estimated mixture objects

    '''
    gmm_estimates = {}
    for device in traffic:
        gmm_estimates[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:
                #print('Estimating {}, direction: {}, parameter {}'.format(device,direction,parameter))
                deviceData = np.array(traffic[device][direction][parameter]).reshape(-1,1)
                gmm_estimates[device][direction][parameter] = \
                mixture.GaussianMixture(n_components=componentNumb, covariance_type='full', random_state=88)
                gmm_estimates[device][direction][parameter].fit(deviceData)
                #print(gmm_estimates[device][direction][parameter].means_)
    return gmm_estimates

def estimateParametersBEM(traffic, componentNumb=5):
    '''
    estimateParametersEM() estimates statistical properties (Gauusian Mixtures) 
    via EM-algorithm for IAT and pktLen parameters for each device/flow and returns
    dict with estimated mixture objects

    '''
    gmm_estimates = {}
    for device in traffic:
        gmm_estimates[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:
                #print('Estimating {}, direction: {}, parameter {}'.format(device,direction,parameter))
                deviceData = np.array(traffic[device][direction][parameter]).reshape(-1,1)
                gmm_estimates[device][direction][parameter] = \
                mixture.BayesianGaussianMixture(n_components=componentNumb, covariance_type='full', random_state=88, weight_concentration_prior=0.01)
                gmm_estimates[device][direction][parameter].fit(deviceData)
                #print(gmm_estimates[device][direction][parameter].means_)
    return gmm_estimates

def estimate_parameters_EM_BIC(traffic, min_samples_to_estimate=15):

    print('Estimating mixtures with EM-algorithm...')
    gmm_estimates = {}
    for device in traffic:
        gmm_estimates[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:
 
                if parameter=='pktLen':
                    compensation = 0
                else:
                    compensation = 1

                if len(traffic[device][direction][parameter]) < min_samples_to_estimate:
                    print('Could not apply EM for {} {} {}'.format(direction,device,parameter))
                    gmm_estimates[device][direction][parameter] = traffic[device][direction][parameter]
                    continue

                #set regularization to depend on max values, for nicier plots and estimation
                reg_cov = 10**(-round(1/max(traffic[device][direction][parameter]))-compensation)                   
                deviceData = np.array(traffic[device][direction][parameter]).reshape(-1,1)
                lowest_bic = np.infty
                bic = []
                for comp in range(1,6):
                    gmm = mixture.GaussianMixture(n_components=comp, covariance_type='full', random_state=88, reg_covar = reg_cov)
                    try:
                        gmm.fit(deviceData)
                        bic.append(gmm.bic(deviceData))
                        if bic[-1] < lowest_bic:
                            lowest_bic = bic[-1]
                            gmm_estimates[device][direction][parameter] = gmm
                    except ValueError:
                        print('Not enough samples for {}. Stopped at {} components'.format(device, comp-1))
                        break 
                print('{} {} {}: selected mixture with {} components ({} max)'.format(direction,device,parameter,np.argmin(bic)+1, len(bic) ))

    return gmm_estimates

def get_dfs_within_percentiles(dfs, percentiles, min_samples_to_estimate):
    
    reduced_dfs = construct_dict_2_layers(dfs)
    for device, direction, df in iterate_dfs_plus(dfs):
        if df.shape[0] < min_samples_to_estimate:
            continue
        
        upper_bound = np.percentile(df['IAT'], percentiles[1])
        lower_bound = np.percentile(df['IAT'], percentiles[0])

        reduced_dfs[device][direction] = df[ (df['IAT']>lower_bound) & (df['IAT']<upper_bound) ]
    return reduced_dfs

def get_data_dict_within_percentiles(data_dict, percentiles, min_samples_to_estimate):
    for device in data_dict:
        for direction in data_dict[device]:
            for parameter in data_dict[device][direction]:
                if len(data_dict[device][direction][parameter]) < min_samples_to_estimate:
                    continue
                data_dict[device][direction][parameter] = get_data_within_percentiles(data_dict[device][direction][parameter], percentiles)
    return data_dict

def fix_const_pktLen(extractedTraffic):

    for device in extractedTraffic:
            for direction in extractedTraffic[device]:
                for parameter in ['pktLen']:
                    if len(extractedTraffic[device][direction][parameter])==0:
                        continue

                    uniqueValues = len(list(Counter(extractedTraffic[device][direction][parameter])))
                    #print(uniqueValues)
                    if uniqueValues==1:
                        extractedTraffic[device][direction][parameter][0]=extractedTraffic[device][direction][parameter][0]+20
                    else:
                        continue

def craft_traffic_features(traffic):

    orig_dfs = get_df_from_traffic(traffic)

    for device, direction, df in iterate_dfs_plus(orig_dfs):
        #apply window smoothing
        #orig_dfs[device][direction]['pkt_var'] = (df['pktLen'] - df['pktLen'].mean())**2
        #orig_dfs[device][direction]['iat_pkt_ratio'] = orig_dfs[device][direction]['IAT']/df['pktLen'] 
        #iat_mean = orig_dfs[device][direction]['IAT'].mean()
        #orig_dfs[device][direction]['iat_var'] = (orig_dfs[device][direction]['IAT'] - iat_mean)**2
        #orig_dfs[device][direction]['iat_log'] = np.log10(orig_dfs[device][direction]['IAT'])
        #orig_dfs[device][direction].index = pd.to_datetime(df['ts'], unit='s') - pd.to_datetime(df['ts'][0], unit='s')
        
        #orig_dfs[device][direction]['window_pkt'] = df['pktLen'].rolling('1S').sum()
        start_index = df.index[0]
        offset_index = start_index + pd.to_timedelta(10, unit='s')
        #orig_dfs[device][direction] = df[ offset_index:]
        #print(orig_dfs[device][direction].head())
        
    return orig_dfs 

def getTrafficFeatures(pcapfile, fileIdent, typeIdent, percentiles=None, min_samples_to_estimate=None):
    identifiers = getAddressList(fileIdent, typeIdent)

    if typeIdent=='flow':
        extractedTraffic = extract_flow_stats(pcapfile, identifiers, min_samples_to_estimate)
    elif typeIdent=='MAC' or typeIdent=='IP':
        extractedTraffic = extractHostStatsDpkt(pcapfile, identifiers, typeIdent, min_samples_to_estimate)
    else:
        exit('wrong identifier type. See help')


    #add more features e.g. IAT, correlation
    traffic_dfs = craft_traffic_features(extractedTraffic)
    if percentiles:
        traffic_dfs = get_dfs_within_percentiles(traffic_dfs, percentiles, min_samples_to_estimate)
    
    fix_const_pktLen(traffic_dfs)
    #len(list(Counter(extractedTraffic[device][direction][parameter])))
    identifiers = getNotEmptyIdentifiers(identifiers, traffic_dfs)
    print('Finished extracting packet properties')
    return traffic_dfs, identifiers

def getTrafficEstimations(extractedTraffic, componentNumb, min_samples_to_estimate=15):
        
    if componentNumb=='auto':
        estimatedParameterEM = estimate_parameters_EM_BIC(extractedTraffic, min_samples_to_estimate)
    else:
        try:
            estimatedParameterEM = estimateParametersEM(extractedTraffic, int(componentNumb))
        except ValueError:
            exit('wrong argument of component number. See help: -h')
    
    kde_estimators = get_KDE_estimators_scipy(extractedTraffic)
    #kde_estimators = get_best_KDE(extractedTraffic)

    trafficExtremeValues = getTrafficExtremeValues(extractedTraffic)
    print('Finished estimating the traffic')

    return estimatedParameterEM, kde_estimators, trafficExtremeValues
 

def saveEstimationsToDisk(args, identifiers, estimations):
    #save estimations to disk
    
    id_s = ['_'.join(identifier.split(' ')[:2]) for identifier in identifiers]
    pcapFileName = get_pcap_filename(args)
    fileToSaveObjects = pcapFileName+'_'+'_'.join(id_s).replace(':','_').replace(' ','__').replace('.','_')+'_'+str(args.n)+'_components'
    save_obj(estimations,fileToSaveObjects)
    print('Saved results to: obj/%s.pkl'%fileToSaveObjects)

def get_best_KDE(traffic):
    
    kde_estimators = {}
    for device in traffic:
        kde_estimators[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:

                deviceData = np.array(traffic[device][direction][parameter]).reshape(-1,1)
                # use grid search cross-validation to optimize the bandwidth
                params = {'bandwidth': np.logspace(-5, 1, 7)}
                grid = GridSearchCV(KernelDensity(), params)
                grid.fit(deviceData)

                print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
                
                kde_estimators[device][direction][parameter] = grid.best_estimator_

    return kde_estimators

def get_KDE(traffic):
    kde_estimators = {}
    for device in traffic:
        kde_estimators[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:
                if parameter=='IAT':
                    bandwidthDef = 0.0002
                else:
                    bandwidthDef = 1

                deviceData = np.array(traffic[device][direction][parameter]).reshape(-1,1)
                kde_estimators[device][direction][parameter] = KernelDensity(bandwidth=bandwidthDef).fit(deviceData)
                
    return kde_estimators

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="pcap file to appoximate flow/device from",required=True)
    parser.add_argument("-i", help="specify identifier type to be read from the file, either 'IP', or 'MAC', or 'flow' ",default='flow')
    #parser.add_argument("-l", help="approximation level, either 'flow' or 'device'",default="device")
    parser.add_argument("-f", help="file with devices or flow identifiers to process: MAC (e.g. xx:xx:xx:xx:xx:xx), IP (e.g. 172.16.0.1), 5-tuple (e.g. TCP 172.16.0.1:4444 172.16.0.2:8888). if 'all' is specified, then every flow within the pcap will be estimated",default="addresses_to_check.txt")
    parser.add_argument("-n", help="estimate with N components. Opt for 'auto' if not sure, although takes more time to estimate", default='auto')
    parser.add_argument("-percentiles", help="specify the lower and upper percentiles to remove possible outliers, e.g. 3,97. Default is 3,97", default='3,97')
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--hist', dest='hist', action='store_true')
    parser.add_argument('--no-hist', dest='hist', action='store_false')
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.add_argument('--save-plot', dest='save_plot', action='store_true')
    parser.set_defaults(plot=False, hist=False, save_plot=False)

    args = parser.parse_args()

    min_samples_to_estimate = 100
    percentiles = (int(args.percentiles.split(',')[0]), int(args.percentiles.split(',')[1]))
    extractedTraffic, identifiers = getTrafficFeatures(args.p, args.f, args.i, percentiles,
                                                        min_samples_to_estimate)

    estimatedParameterEM, kde_estimators, trafficExtremeValues = getTrafficEstimations(extractedTraffic, args.n, min_samples_to_estimate)

    #kde_estimations = get_KDE(extractedTraffic)

    #plot_KDE_traffic(extractedTraffic)

    saveEstimationsToDisk(args, identifiers, \
                        [estimatedParameterEM, trafficExtremeValues])

    printEstimatedParametersEM(estimatedParameterEM)

    if args.plot:
        plot_hist_kde_em(extractedTraffic, kde_estimators, estimatedParameterEM, logScale=False, saveToFile=args.save_plot)

    if args.hist:
        plot_hist(traffic=extractedTraffic, logScale=False)
    
if __name__ == "__main__":
    main()