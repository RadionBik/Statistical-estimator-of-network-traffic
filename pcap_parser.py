import logging
import re
from subprocess import Popen, PIPE
from sys import exit
from typing import Tuple

import dpkt
import numpy as np
import pandas as pd

import utils
import settings

logger = logging.getLogger(__name__)


def _get_flow_list(pcapfile):
    logger.info('Extracting flow identifiers from {}...'.format(pcapfile))
    keys = []
    for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
        eth = dpkt.ethernet.Ethernet(raw)
        ip = eth.data
        # check if the packet is IP, TCP, UDP
        if not isinstance(ip, dpkt.ip.IP):
            continue
        seg = ip.data
        if isinstance(seg, dpkt.tcp.TCP):
            transp_proto = "TCP"
        elif isinstance(seg, dpkt.udp.UDP):
            transp_proto = "UDP"
        else:
            continue

        new_key = (transp_proto, (((ip.src, seg.sport), (ip.dst, seg.dport))))
        if new_key not in keys:
            # logger.info(new_key)
            keys.append(new_key)

    # define set of unique keys
    keys_set = set()
    for key in keys:
        key_set = (key[0], frozenset((key[1][0], key[1][1])))
        # logger.info(key_set)
        keys_set.add(key_set)

    # remove duplicate keys, preserving the order and using the keys_set,
    # workaround to avoid uncertain order of using set-type keys.
    # not the best solution
    identifiers = []
    for orig_key in keys_set:
        found_key = False
        for key in keys:
            if found_key:
                break
            if orig_key == (key[0], frozenset((key[1][0], key[1][1]))):
                identifiers.append(key[0] + ' ' + utils.ip_to_string(list(key[1])[0][0]) + ':' + str(
                    list(key[1])[0][1]) + ' ' + utils.ip_to_string(list(key[1])[1][0]) + ':' + str(
                    list(key[1])[1][1]))
                found_key = True

    return identifiers


def _check_identifiers_type(identifiers):
    identifiersType = {}
    for host in identifiers:
        if utils.is_mac_addr(host):
            identifiersType[host] = utils.TrafficObjects.MAC
        elif utils.is_ip_addr(host):
            identifiersType[host] = utils.TrafficObjects.IP
        elif utils.is_5_tuple(host):
            identifiersType[host] = utils.TrafficObjects.FLOW
        else:
            logger.info('Identifier {} is neither MAC or IP or a flow. Ignoring'.format(host))

    # logger.info(identifiersType)
    return identifiersType


def _extract_flow_stats(pcapfile, flows, min_samples_to_estimate=0, payloadOnly=True):
    logger.info('Extracting flow features from {}...'.format(pcapfile))
    # create the layered dict
    traffic = {}
    if not flows:
        flows = _get_flow_list(pcapfile)

    identifiersType = _check_identifiers_type(flows)
    if type(flows) is str:
        flows = [flows]
    for identifier in flows:
        if identifiersType[identifier] != utils.TrafficObjects.FLOW:
            continue
        # traffic[identifier] = {'from': {'ts': [],'pktLen': [],'IAT': []},
        #                       'to': {'ts': [],'pktLen': [],'IAT': []}}
        traffic[identifier] = {'from': {'ts': [], 'pktLen': []},
                               'to': {'ts': [], 'pktLen': []}}

    for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
        eth = dpkt.ethernet.Ethernet(raw)
        ip = eth.data

        # skip ARP, ICMP, etc.
        if not isinstance(ip, dpkt.ip.IP):
            continue
        if not isinstance(ip.data, dpkt.tcp.TCP) and not isinstance(ip.data, dpkt.udp.UDP):
            continue
        # filter out segments and datagrams without payload (e.g. SYN, SYN/ACK, etc.)
        if payloadOnly and len(ip.data.data) == 0:
            continue

        seg = ip.data
        key_packet_from = (
            isinstance(seg, dpkt.tcp.TCP),
            utils.ip_to_string(ip.src), seg.sport,
            utils.ip_to_string(ip.dst), seg.dport)
        key_packet_to = (
            isinstance(seg, dpkt.tcp.TCP),
            utils.ip_to_string(ip.dst), seg.dport,
            utils.ip_to_string(ip.src), seg.sport)
        for identifier in traffic:
            tup = utils.get_5_tuple_fields(identifier)
            key_ident = (tup['proto'] == 'TCP', tup['ip_s'], int(tup['port_s']), tup['ip_d'], int(tup['port_d']))
            if key_ident == key_packet_from:
                direction = 'from'

            elif key_ident == key_packet_to:
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

    # logger.info the number of packets and remove empty identifiers
    emptyIdentifiers = set()
    for identifier in traffic:
        if len(traffic[identifier]['from']['ts']) < min_samples_to_estimate and \
                len(traffic[identifier]['to']['ts']) < min_samples_to_estimate:
            emptyIdentifiers.add(identifier)

    for identifier in emptyIdentifiers:
        traffic.pop(identifier, None)

    if not traffic:
        exit('Could not find flows with # of packets > {}!'.format(min_samples_to_estimate))

    logger.info('Found the following flows with # of packets > {}:'.format(min_samples_to_estimate))
    for identifier in traffic:
        for direction in traffic[identifier]:
            logger.info('{} pkt number {}: {}'.format(identifier, direction, len(traffic[identifier][direction]['ts'])))

    return traffic


def _extract_host_stats(pcapfile,
                        hosts=None,
                        setIdentifier=utils.TrafficObjects.IP,
                        payloadOnly=True,
                        min_samples_to_estimate=15,
                        max_packet_limit=20000):
    '''
    extractHostsFromPcapDpkt() uses the dpkt lib to extract packet features of the 
    desired hosts.
    'macAddress' searches hosts by MAC addresses if True, or by IP if False
    
    TODO
    1. add IP identifier capability
    2. add tun int capability
    
    '''

    # create the layered dict
    traffic = {}
    if type(hosts) is str:
        hosts = [hosts]

    identifiers_type = _check_identifiers_type(hosts)
    hosts_with_valid_identifiers = []
    for identifier in hosts:

        if identifiers_type[identifier] != setIdentifier:
            continue

        hosts_with_valid_identifiers.append(identifier)
        traffic[identifier] = {'from': {'ts': [], 'pktLen': [], 'IAT': []},
                               'to': {'ts': [], 'pktLen': [], 'IAT': []}}

    logger.info(hosts_with_valid_identifiers)

    packet_counter = 1
    for ts, raw in dpkt.pcap.Reader(open(pcapfile, "rb")):
        if packet_counter % 100000 == 0:
            logger.info('Processed {} packets...'.format(packet_counter))
        eth = dpkt.ethernet.Ethernet(raw)
        ip = eth.data
        packet_counter += 1

        # skip ARP, ICMP, etc.
        if not isinstance(ip, dpkt.ip.IP):
            continue
        # filter out segments and datagrams without payload (e.g. SYN, SYN/ACK, etc.)
        if payloadOnly and len(ip.data.data) == 0:
            continue

        if setIdentifier == utils.TrafficObjects.MAC:
            identifierFrom = utils.mac_addr(eth.src)
            identifierTo = utils.mac_addr(eth.dst)
        elif setIdentifier == utils.TrafficObjects.IP:
            identifierFrom = utils.ip_to_string(eth.data.src)
            identifierTo = utils.ip_to_string(eth.data.dst)
        else:
            raise ValueError('wrong identifier type')

        if identifierFrom in hosts_with_valid_identifiers:
            identifier = identifierFrom
            direction = 'from'
        elif identifierTo in hosts_with_valid_identifiers:
            identifier = identifierTo
            direction = 'to'
        else:
            continue

        if len(traffic[identifier]['from']['ts']) + len(traffic[identifier]['to']['ts']) > max_packet_limit:
            continue

        traffic[identifier][direction]['ts'].append(ts)
        if payloadOnly:
            traffic[identifier][direction]['pktLen'].append(len(ip.data.data))
        else:
            traffic[identifier][direction]['pktLen'].append(len(ip))

    emptyIdentifiers = set()
    for identifier in traffic:
        for direction in traffic[identifier]:
            if len(traffic[identifier][direction]['ts']) < min_samples_to_estimate:
                emptyIdentifiers.add(identifier)

    for identifier in emptyIdentifiers:
        traffic.pop(identifier, None)

    logger.info('Found the following non-empty identifiers:')
    for identifier in traffic:
        for direction in traffic[identifier]:
            logger.info('{} pkt number {}: {}'.format(identifier, direction, len(traffic[identifier][direction]['ts'])))

    return traffic


def _get_dfs_within_percentiles(dfs, percentiles, min_samples_to_estimate):
    reduced_dfs = utils.construct_dict_2_layers(dfs)
    for device, direction, df in utils.iterate_2layer_dict_copy(dfs):
        if df.shape[0] < min_samples_to_estimate:
            continue

        upper_bound = np.percentile(df['IAT'], percentiles[1])
        lower_bound = np.percentile(df['IAT'], percentiles[0])

        reduced_dfs[device][direction] = df[(df['IAT'] > lower_bound) & (df['IAT'] < upper_bound)]
    return reduced_dfs


def _get_data_within_percentiles(device_traffic, percentiles):
    # TODO not used so far
    new_traffic = []
    upper_bound = np.percentile(device_traffic, percentiles[1])
    lower_bound = np.percentile(device_traffic, percentiles[0])

    for entry in device_traffic:
        if entry > upper_bound or entry < lower_bound:
            continue
        new_traffic.append(entry)

    return new_traffic


def _get_data_dict_within_percentiles(data_dict, percentiles, min_samples_to_estimate):
    # TODO not used so far
    for device in data_dict:
        for direction in data_dict[device]:
            for parameter in data_dict[device][direction]:
                if len(data_dict[device][direction][parameter]) < min_samples_to_estimate:
                    continue
                data_dict[device][direction][parameter] = _get_data_within_percentiles(
                    data_dict[device][direction][parameter], percentiles)
    return data_dict


@utils.unpack_2layer_traffic_dict
def _get_df_from_traffic(traffic):
    traffic_df = pd.DataFrame()
    timestamps = pd.Series(traffic['ts'])

    traffic_df['IAT'] = timestamps.diff().fillna(value=0)
    traffic_df['pktLen'] = pd.Series(traffic.get('pktLen'))

    traffic_df.index = pd.to_datetime(timestamps, unit='s') - pd.to_datetime(timestamps[0], unit='s')
    # traffic_df[device][direction]['window_pkt'] = traffic_df[device][direction]['pktLen'].rolling('1S').sum()

    return traffic_df


def _get_address_list(file, type_ident: utils.TrafficObjects):
    """
    _get_address_list() returns a list with identifiers to process, empty if file is None
    """
    logger.info(f'reading list of addresses...')
    address_list = []
    if file is not None and type_ident != utils.TrafficObjects.FLOW:
        with open(file, 'r') as f:
            logger.info('Reading identifiers from the {} file...'.format(file))
            for line in f:
                if line[0] == '#':
                    continue
                # logger.info(line, end='')
                address_list.append(line.rstrip())
    return address_list


def get_traffic_features(pcapfile: str,
                         type_of_identifier: utils.TrafficObjects,
                         identifiers: list = None,
                         percentiles: Tuple[int, int] = None,
                         min_samples_to_estimate: int = None):
    if type_of_identifier is utils.TrafficObjects.FLOW:
        extracted_traffic = _extract_flow_stats(pcapfile,
                                                flows=identifiers,
                                                min_samples_to_estimate=min_samples_to_estimate)
    elif type_of_identifier is utils.TrafficObjects.MAC or \
            type_of_identifier is utils.TrafficObjects.IP:
        extracted_traffic = _extract_host_stats(pcapfile,
                                                hosts=identifiers,
                                                setIdentifier=type_of_identifier,
                                                min_samples_to_estimate=min_samples_to_estimate)
    else:
        raise TypeError('wrong identifier type. See help')

    # add more features e.g. IAT, correlation
    traffic_dfs = _get_df_from_traffic(extracted_traffic)
    if percentiles:
        traffic_dfs = _get_dfs_within_percentiles(traffic_dfs, percentiles, min_samples_to_estimate)

    # len(list(Counter(extracted_traffic[device][direction][parameter])))
    logger.info('Finished extracting packet properties')
    return traffic_dfs, identifiers
