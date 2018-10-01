#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re
import argparse
from subprocess import Popen, PIPE
import cProfile
import dpkt
from dpkt.compat import compat_ord
import pcap
import pandas as ps
import numpy as np
import sklearn
import socket
import copy
import matplotlib.pyplot as plt
import pickle
import os
from sklearn import mixture

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


def ip_from_string(ips):
    '''
        Convert symbolic IP-address into a 4-byte string
        Args:
            ips - IP-address as a string (e.g.: '10.0.0.1')
        returns:
            a 4-byte string
    '''
    return b''.join([bytes([int(n)]) for n in ips.split('.')])


def mac_addr(address):
    """Convert a MAC address to a readable/printable string

       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return ':'.join('%02x' % compat_ord(b) for b in address)


def parseVoIPpacker(pcapfile, filter):
    for ts, raw in ppcap.Reader(filename=pcapfile):
        eth = ethernet.Ethernet(raw)

        # create the keys for IP UDP/TCP flows
        if eth[ip.IP] is not None:
            # if eth[tcp.TCP] is not None:
            #    continue
                # key = ('tcp', frozenset(((eth.ip.src_s, eth.ip.tcp.sport),(eth.ip.dst_s, eth.ip.tcp.dport))))
            if eth[udp.UDP] is not None:
                # and ((eth.ip.udp.sport in portsOfInterest) or (eth.ip.udp.dport in portsOfInterest)):
                if ((eth.ip.src_s in filter['IP']) or (eth.ip.dst_s in filter['IP'])):
                    if ((eth.ip.udp.sport in filter['port']) or (eth.ip.udp.dport in filter['port'])):
                        key = (ts, 'udp', frozenset(
                            ((eth.ip.src_s, eth.ip.udp.sport), (eth.ip.dst_s, eth.ip.udp.dport))))
                        print(key)
        else:
            continue


def get_IAT(TS):

    iteration = 0
    IAT = [0]
    for ts in TS:
        if iteration == 0:
            tempIAT = ts
            iteration = iteration + 1
        else:
            IAT.append(ts - tempIAT)
            tempIAT = ts
    return IAT


def parseVoIP(pcapfile, filter=None, isTunInt=False, manualFiltering=False):

    pipe = Popen(["./ndpiReader", "-i", pcapfile, "-v2"], stdout=PIPE)
    raw = pipe.communicate()[0].decode("utf-8")

    reg = re.compile(
        r'(UDP) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) <?->? (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5}) \[proto: [\d+\.]*\d+\/(RTP)*\]')

    if not manualFiltering:

        filter = {
            'IPserver': [],
            'IPclient': [],
            'port': [],
            'length': []
        }

        print('Found the following RTP flows:')
        for captures in re.findall(reg, raw):
            print(captures)
            transp_proto, ip1, port1, ip2, port2, app_proto = captures
            filter['IPserver'].append(ip1)
            filter['IPclient'].append(ip2)
            filter['port'].append(int(port1))
            filter['port'].append(int(port2))

    clientToServerPkts = {
        'ts': [],
        'pktLen': [],
        'IAT': []
    }

    serverToClientPkts = copy.deepcopy(clientToServerPkts)

    pktNum = 0
    t = 1
    iteration = 0
    for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
        if (t == 1):
            firstTime = ts
            t = t-1
        if (isTunInt):
            ip = dpkt.ip.IP(raw)
        else:
            eth = dpkt.ethernet.Ethernet(raw)
            ip = eth.data
        # check if the packet is IP, TCP, UDP
        if not isinstance(ip, dpkt.ip.IP):
            continue
        seg = ip.data
        if isinstance(seg, dpkt.udp.UDP):
            timeFromStart = ts - firstTime
            if (ip_to_string(ip.src) in filter['IPserver']) or (ip_to_string(ip.dst) in filter['IPserver']):
                if ((ip.data.sport in filter['port']) and (ip.data.dport in filter['port'])) and len(raw) not in filter['length']:
                    if ip_to_string(ip.src) == filter['IPclient'][0]:
                        clientToServerPkts['ts'].append(timeFromStart)
                        clientToServerPkts['pktLen'].append(len(raw))

                    else:
                        serverToClientPkts['ts'].append(timeFromStart)
                        serverToClientPkts['pktLen'].append(len(raw))

                    pktNum = pktNum+1

        else:
            continue

    clientToServerPkts['IAT'] = get_IAT(clientToServerPkts['ts'])
    serverToClientPkts['IAT'] = get_IAT(serverToClientPkts['ts'])
    return clientToServerPkts, serverToClientPkts


def extractPacketStatsPerDevice(pcapfile, mac_list):

    plain_mac_list = []
    # create traffic dicts from the MAC dict
    packetsFrom = {}
    packetsTo = {}
    for device_group in mac_list:
        packetsFrom[device_group] = {}
        packetsTo[device_group] = {}
        for device in mac_list[device_group]:
            packetsTo[device_group][device] = {
                'ts': [], 'pktLen': [],  'IAT': []}
            packetsFrom[device_group][device] = {
                'ts': [], 'pktLen': [],  'IAT': []}
            plain_mac_list.append(device)

    for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
        eth = dpkt.ethernet.Ethernet(raw)
        for device_group in mac_list:
            # print(device_group)
            if mac_addr(eth.src) in list(packetsFrom[device_group].keys()):
                packetsFrom[device_group][mac_addr(eth.src)]['ts'].append(ts)
                packetsFrom[device_group][mac_addr(
                    eth.src)]['pktLen'].append(len(eth.data))

            elif mac_addr(eth.dst) in list(packetsTo[device_group].keys()):
                packetsTo[device_group][mac_addr(eth.dst)]['ts'].append(ts)
                packetsTo[device_group][mac_addr(
                    eth.dst)]['pktLen'].append(len(eth.data))
            else:
                continue

    for device_group in packetsTo:
        for device in packetsTo[device_group]:
            packetsFrom[device_group][device]['IAT'] = get_IAT(
                packetsFrom[device_group][device]['ts'])
            packetsTo[device_group][device]['IAT'] = get_IAT(
                packetsTo[device_group][device]['ts'])

    return packetsFrom, packetsTo


def plot_IAT(list, title, fig_properties):
    f1 = plt.figure(figsize=fig_properties['size'])
    plt.hist(
        list, bins=fig_properties['bins'])  # range=fig_properties['range'])
    plt.title(title)
    plt.xlabel('IAT, s')
    plt.ylabel('number')
    plt.grid(True)
    f1.show()


def plot_PL(list, title, fig_properties):
    f2 = plt.figure(figsize=fig_properties['size'])
    plt.hist(list, bins=fig_properties['bins'])
    plt.title(title)
    plt.xlabel('bytes')
    plt.ylabel('number')
    plt.grid(True)
    f2.show()


def plotPSOfTime(dict, fig_properties, title):
    f3 = plt.figure(figsize=fig_properties['size'])
    plt.plot(dict['ts'][::fig_properties['sampling']],
             dict['pktLen'][::fig_properties['sampling']])
    plt.title(title)
    plt.ylabel('Packet size, bytes')
    plt.xlabel('time, s')
    plt.grid(True)
    f3.show()


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def groupDevicesByType(packets):
    groupedPackets = {}
    for device_group in packets:
        groupedPackets[device_group] = {'ts': [], 'pktLen': [],  'IAT': []}
        for device in packets[device_group]:
            groupedPackets[device_group]['ts'].append(
                packets[device_group][device]['ts'])
            groupedPackets[device_group]['pktLen'].append(
                packets[device_group][device]['pktLen'])
            groupedPackets[device_group]['IAT'].append(
                packets[device_group][device]['IAT'])

    return groupedPackets


def groupDevicesAllTogether(packets):
    groupedPackets = {'ts': [], 'pktLen': [],  'IAT': []}
    for device_group in packets:
        for device in packets[device_group]:
            groupedPackets['ts'].append(
                packets[device_group][device]['ts'][:50])
            groupedPackets['pktLen'].append(
                packets[device_group][device]['pktLen'][:50])
            groupedPackets['IAT'].append(
                packets[device_group][device]['IAT'][:50])

    return groupedPackets


def unfoldDevices(packets):

    groupedPackets = {}
    for device_group in packets:
        for device in packets[device_group]:
            groupedPackets[device] = {'ts': [], 'pktLen': [],  'IAT': []}
            groupedPackets[device]['ts'] = packets[device_group][device]['ts'][:50]
            groupedPackets[device]['pktLen'] = packets[device_group][device]['pktLen'][:50]
            groupedPackets[device]['IAT'] = packets[device_group][device]['IAT'][:50]

    return groupedPackets


def getHistForEveryDevice(packets, parameter):
    
    detalization = 20
    histGen = np.zeros(detalization-1)
    hist_properties = {}
    if parameter is 'pktLen':
        hist_properties['range'] = (0, 1500)
        hist_properties['bins'] = np.linspace(0,1500,detalization)

    elif parameter is 'IAT':
        hist_properties['range'] = None#(0, 2000)
        #hist_properties['bins'] = detalization-1
        hist_properties['bins'] = np.logspace(np.log10(0.00001), np.log10(2000), detalization)
        #hist_properties['bins'] =  np.linspace(0,1000,detalization)
    else:
        print('Are you sure?')

    
    for device_group in packets:
        for device in packets[device_group]:
            hist, bins = np.histogram(packets[device_group][device][parameter], bins=hist_properties['bins'], range=hist_properties['range'])
            histGen = np.add(histGen, hist)
            print(device_group, device)
            print(hist)
            #print(bins)
    
    
def plotSummedParameter(packets, parameter, fig_properties, title, saveToFile, logScale):
    
    if parameter is 'pktLen':
        fig_properties['range'] = (0, 1500)
        fig_properties['bins'] = np.linspace(0,1500,100)

    elif parameter is 'IAT':
        fig_properties['range'] = None
        if logScale:
            fig_properties['bins'] = np.logspace(np.log10(0.00001), np.log10(2000), 100)
        else:
            fig_properties['bins'] = np.linspace(0,1.5,100)
    else:
        print('Are you sure?')


    summedPackets = []
    summedPackets_ts = []
    for device_group in packets:
        for device in packets[device_group]:
            if parameter is 'IAT':
                summedPackets_ts = summedPackets_ts + packets[device_group][device]['ts']
            else:
                summedPackets = summedPackets + packets[device_group][device][parameter]

    if parameter is 'IAT':
        summedPackets = get_IAT(summedPackets_ts)
    print(len(summedPackets))
    histGen, binsGen = np.histogram(summedPackets, bins=fig_properties['bins'], range=fig_properties['range'])
    
    print(histGen)
    
    f2 = plt.figure(figsize=fig_properties['size'])

    plt.hist(summedPackets, bins=fig_properties['bins'], range=fig_properties['range'])#, density=True)
    if parameter is 'IAT' and logScale:
        plt.gca().set_xscale("log")
    plt.title(title.split('|')[1])
    plt.xlabel(title.split('|')[0])
    plt.ylabel('Number')
    plt.xticks(fontsize = fig_properties['font'])
    plt.yticks(fontsize = fig_properties['font'])
    plt.grid(True)
    
    if saveToFile:
        if logScale and parameter is 'IAT':
            filename = 'trafficFigures'+os.sep+title.replace('|','_').replace(' ','')+'_log'+'.svg'
        else:
            filename = 'trafficFigures'+os.sep+title.replace('|','_').replace(' ','')+'.svg'
        plt.savefig(filename)
    else:
        f2.show()


def plotParameterForEveryDevice(packets, parameter, fig_properties, title, saveToFile, logScale):

    if parameter is 'pktLen':
        fig_properties['range'] = (0, 1500)
        fig_properties['bins'] = np.linspace(0,1500,100)

    elif parameter is 'IAT':
        fig_properties['range'] = None
        if logScale:
            fig_properties['bins'] = np.logspace(np.log10(0.00001), np.log10(2000), 100)
        else:
            fig_properties['bins'] = np.linspace(0,1.5,100)
    else:
        print('Are you sure?')
        
    for device_group in packets:
        for device in packets[device_group]:
            f2 = plt.figure(figsize=fig_properties['size'])

            plt.hist(packets[device_group][device][parameter],
                     bins=fig_properties['bins'], range=fig_properties['range'], density=False)
            if parameter is 'IAT' and logScale:
                plt.gca().set_xscale("log")
            plt.legend([device])
            plt.title(title.split('|')[1]+" ("+device_group+": "+device+" )")
            plt.xlabel(title.split('|')[0])
            plt.ylabel('Number')
            plt.grid(True)
            
            if saveToFile:
                filename = 'trafficFigures'+os.sep+'byDevices'+os.sep+device_group+'_'+device+'_'+title.replace('|','_').replace(' ','')
                if logScale and parameter is 'IAT':
                    filename = filename+'_log'+'.svg'
                else:
                    filename = filename+'.svg'
                plt.savefig(filename)
            else:
                f2.show() 


def plotParameterByGroups(packets, parameter, fig_properties, title):

    if parameter is 'pktLen':
        fig_properties['range'] = (0, 1500)
        fig_properties['bins'] = list(range(0, 1500, 25))

    elif parameter is 'IAT':
        fig_properties['range'] = None
        fig_properties['bins'] = np.logspace(
            np.log10(0.0001), np.log10(1000.0), 100)
    else:
        print('Are you sure?')

    for device_group in packets:
        deviceList = []
        parameterToPlot = []
        for device in packets[device_group]:
            deviceList.append(device)
            parameterToPlot.append(packets[device_group][device][parameter])
        f2 = plt.figure(figsize=fig_properties['size'])
        plt.hist(parameterToPlot,
                 bins=fig_properties['bins'], range=fig_properties['range'])
        plt.legend(deviceList)
        plt.title(title.split('|')[1]+" ( the "+device_group+" group)")
        plt.xlabel(title.split('|')[0])
        plt.ylabel('Number')
        plt.grid(True)
        f2.show()
        input()


def plotParametersForAllDevices(packets, parameter, fig_properties, title):
    # groupedPacketsByType = groupDevicesByType(packets)
    # groupedPacketsAllTogether = groupDevicesAllTogether(packets)
    # unfoldedDevices = unfoldDevices(packets)

    deviceList = []
    parameterToPlot = []
    for device_group in packets:
        for device in packets[device_group]:
            deviceList.append(device)
            parameterToPlot.append(packets[device_group][device][parameter])
    f2 = plt.figure(figsize=fig_properties['size'])
    plt.hist(parameterToPlot)
    plt.legend(deviceList)
    plt.title(title.split('|')[1])
    plt.xlabel(title.split('|')[0])
    plt.ylabel('Number')
    plt.grid(True)
    f2.show()
    input()

def estimateParametersEM(traffic, componentNumb=3, componentWeightThreshold=0.05):

    gmm = mixture.GaussianMixture(n_components=componentNumb, covariance_type='full', random_state=88)
    estimatedParameters = {}
    for direction in traffic:
        packets = traffic[direction]
        for parameter in ['IAT', 'pktLen']:
            with open('log'+os.sep+'estimateParametersEM_1D_'+direction+'_'+parameter+'.txt', 'w') as fileToSave:
                estimatedParameters[direction] = {}
                print(gmm.get_params(), file = fileToSave)
                for device_group in sorted(packets):
                    estimatedParameters[direction][device_group] = {}
                    deviceList = []
                    for device in sorted(packets[device_group]):
                        if len(packets[device_group][device][parameter]) < 200:
                            continue
                        deviceList.append(device)
                        deviceData = np.array(packets[device_group][device][parameter]).reshape(-1,1)
                        print('{}, {} \t № of pkts: {}'.format(device_group,device,len(deviceData)), file = fileToSave)
                        gmm.fit(deviceData)

                        print('\tWeight\t Mean \t StdDev', file = fileToSave)
                        estimatedParameters[direction][device_group][device] = {'weight': [], 'mean' : [], 'stdDev' : [], 'number' : 0, 'estimator' : gmm}
                        for parSet in zip(gmm. weights_, gmm.means_, np.sqrt(gmm.covariances_)):
                            if np.asscalar(parSet[0]) > componentWeightThreshold:
                                estimatedParameters[direction][device_group][device]['weight'].append(np.asscalar(parSet[0]))
                                estimatedParameters[direction][device_group][device]['mean'].append(np.asscalar(parSet[1]))
                                estimatedParameters[direction][device_group][device]['stdDev'].append(np.asscalar(parSet[2]))
                                estimatedParameters[direction][device_group][device]['number'] = len(deviceData)
                                print('{0:-12.3f} {1:-12.6f} {2:-12.6f}'.format(np.asscalar(parSet[0]), np.asscalar(parSet[1]), \
                                np.asscalar(parSet[2]) ), file = fileToSave)
                        print('\n', file = fileToSave)
                        #print(gmm.predict(deviceData))
                        #plt.plot()
                        #plt.show()
                        #input()
    return estimatedParameters
    
def estimateParametersEM_2D(traffic, componentNumb=3, componentWeightThreshold=0.05):
    gmm = mixture.GaussianMixture(n_components=componentNumb, covariance_type='full', random_state=88)
    estimatedParameters = {}
    n_samples = 50
    
    for direction in traffic:
        packets = traffic[direction]
        with open('log'+os.sep+'estimateParametersEM_2D_'+direction+'.txt', 'w') as fileToSave:
            for device_group in sorted(packets):
                estimatedParameters[device_group] = {}
                for device in sorted(packets[device_group]):
                    if len(packets[device_group][device]['pktLen']) < 200:
                        continue
                    deviceData = np.transpose( [ packets[device_group][device]['IAT'] , packets[device_group][device]['pktLen']] )
                    print('{}, {} \t № of pkts: {}'.format(device_group,device,len(deviceData)), file=fileToSave)
                    gmm.fit(deviceData)
                    for parSet in zip(gmm. weights_, gmm.means_, gmm.covariances_):
                        if np.asscalar(parSet[0]) > componentWeightThreshold:
                            print('{0:-6.3f}, PL: ({2:-6.6f}, {4:-6.6f}), IAT: ({1:-6.6f}, {3:-6.6f}) ; '.format(parSet[0], \
                            parSet[1][0], parSet[1][1], np.sqrt(np.diagonal(parSet[2]))[0], np.sqrt(np.diagonal(parSet[2]))[1]  ), file=fileToSave)
                    #print('\tWeight\t Mean \t StdDev')
                    #estimatedParameters[device_group][device] = {'weight': [], 'mean' : [], 'stdDev' : []}
                    #print(estimatedParameter)
    

def drawMixture(mixtures):
    print('hello')
    
def main():

    MAC = {'hubs': ['d0:52:a8:00:67:5e', '44:65:0d:56:cc:d3'],
           'cameras': ['70:ee:50:18:34:43', 'f4:f2:6d:93:51:f1', '00:16:6c:ab:6b:88',
                       '30:8c:fb:2f:e4:b2', '00:62:6e:51:27:2e', 'e8:ab:fa:19:de:4f',
                       '00:24:e4:11:18:a8'],
           'switches': ['ec:1a:59:79:f4:89', '50:c7:bf:00:56:39', '74:c6:3b:29:d7:1d', 'ec:1a:59:83:28:11'],
           'airQuality': ['18:b4:30:25:be:e4', '70:ee:50:03:b8:ac'],
           'healthcare': ['00:24:e4:1b:6f:96', '74:6a:89:00:2e:25', '00:24:e4:20:28:c6'],
           'lightBulb': ['d0:73:d5:01:83:08'],
           'appliances': ['18:b7:9e:02:20:44',  'e0:76:d0:33:bb:85', '70:5a:0f:e4:9b:c0']
           }

    fig_properties = {
        'size': (5, 4),
        'bins': 100,
        'range': (0, 0.1),
        'sampling': 1,
        'font' : 8
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('file.pcap', nargs='?',
                        default='iotMerged.pcap', help="pcap file")
    args = parser.parse_args()
    try:
        pcapFileName = args.file.split(
            '/')[len(args.file.split('/'))-1].split('.')[0]
    except:
        pcapFileName = 'iotMerged'
    # print(pcapFileName)
    filter = {
        'IPserver': ['192.168.0.105'],
        'IPclient': ['192.168.0.102'],
        'port': [26454, 18826],
        'length': []
    }

    isTunInt = False
    useManualFilter = False
    plotFigures = False
    freshPcap = False
    # clientToServerPkts, serverToClientPkts = parseVoIP(
    #    args.file, filter, isTunInt, useManualFilter)

    if freshPcap:
        trafficFrom, trafficTo = extractPacketStatsPerDevice(args.file, MAC)
        save_obj(trafficFrom, 'trafficFrom_'+pcapFileName)
        save_obj(trafficTo, 'trafficTo_'+pcapFileName)
    else:
        trafficTo = load_obj('trafficTo_'+pcapFileName)
        trafficFrom = load_obj('trafficFrom_'+pcapFileName)

    if freshPcap:
        traffic = {'from' : trafficFrom, 'to' : trafficTo}
        #estimateParametersEM_2D(traffic, 5)
        estimatedMixtures = estimateParametersEM(traffic, 5)
        save_obj(estimatedMixtures, 'estimationsEM_1D_'+pcapFileName)
    else:
        estimatedMixtures = load_obj('estimationsEM_1D_'+pcapFileName)

    #print(estimatedMixtures)
    drawMixture(estimatedMixtures)
    
    '''
    estimatedParametersFrom = {}
    estimatedParametersTo = {}
    for parameter in ['IAT', 'pktLen']: 
        estimatedParametersFrom[parameter] = estimateParametersEM(trafficFrom, parameter, 5)
        estimatedParametersTo[parameter] = estimateParametersEM(trafficTo, parameter, 5)

    #print(estimatedIATfrom)
    
    ################################################
    #   PLOTS FOR INDIVIDUAL DEVICES
    ################################################
    plotParameterForEveryDevice(trafficFrom, 'IAT', fig_properties, 'IAT, sec | Traffic FROM the Device', True, logScale=True)
    plotParameterForEveryDevice(trafficTo, 'IAT', fig_properties, 'IAT, sec | Traffic TO the Device', True, logScale=True)
    
    plotParameterForEveryDevice(trafficFrom, 'IAT', fig_properties, 'IAT, sec | Traffic FROM the Device', True, logScale=False)
    plotParameterForEveryDevice(trafficTo, 'IAT', fig_properties, 'IAT, sec | Traffic TO the Device', True, logScale=False)
    
    plotParameterForEveryDevice(trafficFrom, 'pktLen', fig_properties, 'PL, bytes | Traffic FROM the Device', True, logScale=False)
    plotParameterForEveryDevice(trafficTo, 'pktLen', fig_properties, 'PL, bytes | Traffic TO the Device', True, logScale=False)
    
    ################################################
    #   GENERAL PLOT FOR ALL DEVICES
    ################################################
    plotSummedParameter(trafficFrom, 'IAT', fig_properties, 'IAT, sec | Traffic FROM the Devices', True,logScale = False)
    plotSummedParameter(trafficTo, 'IAT', fig_properties, 'IAT, sec | Traffic TO the Devices', True,logScale = False)
    logScale = False
    plotSummedParameter(trafficFrom, 'IAT', fig_properties, 'IAT, sec | Traffic FROM the Devices', True, logScale = True)
    plotSummedParameter(trafficTo, 'IAT', fig_properties, 'IAT, sec | Traffic TO the Devices', True, logScale = True)

    plotSummedParameter(trafficFrom, 'pktLen', fig_properties, 'PL, bytes | Traffic FROM the Devices', True,logScale = False)
    plotSummedParameter(trafficTo, 'pktLen', fig_properties, 'PL, bytes | Traffic TO the Devices', True,logScale = False)
    # plt.show()
    ################################################
    '''
    #input()

    if plotFigures:
        plot_IAT(clientToServerPkts['IAT'],
                 'Inter-Arrival Time Client -> Server', fig_properties)
        plot_IAT(serverToClientPkts['IAT'],
                 'Inter-Arrival Time Server -> Client', fig_properties)
        plot_PL(clientToServerPkts['pktLen'],
                'Packet Length Client -> Server', fig_properties)
        plot_PL(serverToClientPkts['pktLen'],
                'Packet Length Server -> Client', fig_properties)

        plotPSOfTime(clientToServerPkts, fig_properties,
                     'PS of time Client -> Server')

        plotPSOfTime(serverToClientPkts, fig_properties,
                     'PS of time Server -> Client')

        input()


if __name__ == "__main__":
    main()
