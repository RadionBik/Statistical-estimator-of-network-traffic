import pickle
import re
import socket
import functools

from collections import defaultdict
import numpy as np
import pandas as pd
from dpkt.compat import compat_ord
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cProfile


def get_df_from_traffic(traffic):
    traffic_df = defaultdict(dict)
    for device in traffic:
        for direction in traffic[device]:
            traffic_df[device][direction] = pd.DataFrame()
            timestamps = pd.Series(traffic[device][direction]['ts'])
            traffic_df[device][direction]['IAT'] = timestamps.diff().fillna(value=0)

            traffic_df[device][direction]['pktLen'] = pd.Series(traffic[device][direction]['pktLen'])

            traffic_df[device][direction].index = pd.to_datetime(timestamps, unit='s') - pd.to_datetime(timestamps[0],
                                                                                                        unit='s')

            # traffic_df[device][direction]['window_pkt'] = traffic_df[device][direction]['pktLen'].rolling('1S').sum()

    return traffic_df


def _unpack_2layer_args(args, device, direction):
    unpacked_args = []
    for arg in args:
        if isinstance(arg, dict) and device in arg and direction in arg[device]:
            arg = arg[device][direction]
        unpacked_args.append(arg)
    return unpacked_args


def unpack_2layer_traffic_dict(func):
    @functools.wraps(func)
    def wrapper(traffic_dict: dict, *args, **kwargs):
        new_dfs1 = defaultdict(dict)
        new_dfs2 = defaultdict(dict)
        two_results = False
        for device, dev_df in traffic_dict.items():
            for direction, direct_df in dev_df.items():
                unpacked_args = _unpack_2layer_args(args, device, direction)
                result = func(direct_df, *unpacked_args, **kwargs)
                if result is not None:
                    new_dfs1[device][direction] = result
                if isinstance(result, tuple) and len(result) == 2:
                    two_results = True
                    new_dfs1[device][direction] = result[0]
                    new_dfs2[device][direction] = result[1]
        if two_results:
            return new_dfs1, new_dfs2
        else:
            return new_dfs1

    return wrapper


def unpack_3layer_traffic_dict(func):
    @functools.wraps(func)
    def wrapper(traffic_df: dict, *args, **kwargs):
        new_dfs = defaultdict(dict)
        for device, dev_df in traffic_df.items():
            new_dfs[device] = defaultdict(dict)
            for direction, direct_df in dev_df.items():
                for parameter, param_df in direct_df.items():
                    result = func(param_df, *args, **kwargs)
                    if result is not None:
                        new_dfs[device][direction][parameter] = result
        return new_dfs

    return wrapper


def iterate_traffic_dict(traff_dict):
    for device in traff_dict:
        for direction in traff_dict[device]:
            dat = traff_dict[device][direction]
            yield device, direction, dat


def iterate_traffic_3_layers(traff_dict):
    for device in traff_dict:
        for direction in traff_dict[device]:
            for parameter in traff_dict[device][direction]:
                dat = traff_dict[device][direction][parameter]
                yield device, direction, parameter, dat


def iterate_dfs(traffic):
    for device in traffic:
        for direction in traffic[device]:
            df = traffic[device][direction].copy()
            yield df


def iterate_dfs_plus(traffic):
    for device in traffic:
        for direction in traffic[device]:
            df = traffic[device][direction].copy()
            yield device, direction, df


def profile(func):
    """Decorator for run function profile"""

    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result

    return wrapper


def find_max_parameters(traffic_dfs):
    max_params = {'pktLen': 0, 'IAT': 0}

    for device in traffic_dfs:
        for parameter in ['pktLen', 'IAT']:
            max_params[parameter] = max(
                [max(traffic_dfs[device]['from'][parameter]), max(traffic_dfs[device]['to'][parameter])])

    return max_params


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


def mod_addr(mac):
    '''
    replaces : and . with _ for addresses to enable saving to a disk
    '''
    return mac.replace(':', '_').replace('.', '_').replace(' ', '_')


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
    if re.match(
            "6553[0-6]|655[0-2][0-9]|65[0-4][0-9]{2}|6[0-4][0-9]{3}|[1-5][0-9]{4}|0?[1-9][0-9]{3}|0?0?[1-9][0-9]{"
            "2}|0?0?0?[1-9][0-9]|0{0,4}[1-9]",
            string):
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

    if re.match("udp|tcp", tup['proto'].lower()) and is_ip_port(tup['port_s']) and is_ip_port(
            tup['port_d']) and is_ip_addr(tup['ip_s']) and is_ip_addr(tup['ip_d']):
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


def construct_dict_2_layers(ref_dict):
    new_dict = defaultdict(dict)
    for device in ref_dict:
        for direction in ref_dict[device]:
            new_dict[device][direction] = None

    return new_dict


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
                if parameter == 'ts':
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
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def getAddressList(file, typeIdent):
    '''
    getAddressList() returns a list with identifiers to process, empty if 'all'
    specified
    '''

    addressList = []
    if file != 'all' and typeIdent != 'flow':
        with open(file, 'r') as f:
            print('Reading identifiers from the {} file...'.format(file))
            for line in f:
                if line[0] == '#':
                    continue
                # print(line, end='')
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
    return args.p.split('/')[len(args.p.split('/')) - 1].split('.')[0]


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


def convert_arrays_to_dfs(samples):
    gener_dfs = construct_dict_2_layers(samples)
    for device, direction, df in iterate_traffic_dict(samples):
        gener_dfs[device][direction] = pd.DataFrame(df, columns=['pktLen', 'IAT'])

    return gener_dfs


@unpack_2layer_traffic_dict
def normalize_dfs(df, parameters=None, std_scaler=True):
    if std_scaler:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    if parameters:
        df = df[parameters]

    if 'pktLen' in df.columns:
        df['pktLen'] = df['pktLen'].astype(float, copy=False)

    scaler.fit(df)
    norm_traffic = pd.DataFrame(scaler.transform(df),
                                columns=df.columns)

    return norm_traffic, scaler
