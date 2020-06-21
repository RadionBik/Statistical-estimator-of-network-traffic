import cProfile
import functools
import logging
import pickle
import re
import socket
from collections import defaultdict
from enum import Enum

import pandas as pd
from dpkt.compat import compat_ord
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import settings

logger = logging.getLogger(__name__)


def _unpack_2layer_args(args, device, direction):
    unpacked_args = []
    for arg in args:
        if isinstance(arg, dict) and device in arg and direction in arg[device]:
            arg = arg[device][direction]
        unpacked_args.append(arg)
    return unpacked_args


def unpack_2layer_traffic_dict(func):
    """
    The decorator that simplifies accessing values within (many) and returning (<=2) traffic_dicts
    a 2-layer traffic dict has the following structure of the layers:
    1. device or flow ID
    2. direction {'to', 'from'}
    :param func:
    :return:
    """

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


def iterate_2layer_dict(traff_dict):
    for device in traff_dict:
        for direction in traff_dict[device]:
            dat = traff_dict[device][direction]
            yield device, direction, dat


def iterate_2layer_dict_copy(traffic):
    for device in traffic:
        for direction in traffic[device]:
            df = traffic[device][direction].copy()
            yield device, direction, df


def iterate_3layer_dict(traff_dict):
    for device in traff_dict:
        for direction in traff_dict[device]:
            for parameter in traff_dict[device][direction]:
                dat = traff_dict[device][direction][parameter]
                yield device, direction, parameter, dat


def profile(func):
    """Decorator for run function profile"""

    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result

    return wrapper


def mod_addr(mac):
    """
    replaces : and . with _ for addresses to enable saving to a disk
    """
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
        tuple_dict = {'proto': string.split(' ')[0],
                      'ip_s': string.split(' ')[1].split(':')[0],
                      'port_s': string.split(' ')[1].split(':')[1],
                      'ip_d': string.split(' ')[2].split(':')[0],
                      'port_d': string.split(' ')[2].split(':')[1]}

        return tuple_dict

    except IndexError:
        logger.info('Catched either empty or incorrect lines. Ignoring.')


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


def save_obj(obj, name):
    """
    save_obj() saves python object to the file inside 'obj' directory
    """
    dest_path = settings.BASE_DIR / f'obj/{name}.pkl'
    with open(dest_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    logger.info(f'pickled obj to {dest_path}')
    return dest_path.as_posix()


def load_obj(name):
    """
    load_obj() loads python object from the file inside 'obj' directory
    """
    load_path = settings.BASE_DIR / f'obj/{name}.pkl'
    with open(load_path, 'rb') as f:
        obj = pickle.load(f)
        logger.info(f'loaded obj from {load_path}')
        return obj, load_path.as_posix()


class TrafficObjects(Enum):
    MAC = 'MAC'
    IP = 'IP'
    FLOW = 'flow'


def get_traffic_extreme_values(traffic):
    """
    get_traffic_extreme_values() extracts extreme values i.e. max and
    min values of 'IAT' and 'pktLen' from traffic dict and returns
    dict with them
    """

    extreme_values = defaultdict(dict)
    for device in traffic:
        for direction in traffic[device]:
            extreme_values[device][direction] = {'pktLen': {}, 'IAT': {}}
            for parameter in ['pktLen', 'IAT']:
                extreme_values[device][direction][parameter] = {
                    'max': 0, 'min': 0}

                try:
                    extreme_values[device][direction][parameter]['min'] = min(
                        traffic[device][direction][parameter])
                except ValueError:
                    extreme_values[device][direction][parameter]['min'] = 0
                try:
                    extreme_values[device][direction][parameter]['max'] = max(
                        traffic[device][direction][parameter])
                except ValueError:
                    extreme_values[device][direction][parameter]['max'] = 0

    return extreme_values


@unpack_2layer_traffic_dict
def normalize_dfs(df, scaler=None, parameters=None, std_scaler=False):
    if parameters:
        df = df[parameters]

    if 'pktLen' in df.columns:
        df.loc[:, 'pktLen'] = df['pktLen'].astype(float, copy=False)

    if not scaler:
        scaler = StandardScaler() if std_scaler else MinMaxScaler()
        transformed = scaler.fit_transform(df)
    else:
        transformed = scaler.transform(df)

    norm_traffic = pd.DataFrame(transformed, columns=df.columns)

    return norm_traffic, scaler


def split_train_test_dfs(traffic_dfs, test_size):
    train_dfs = construct_dict_2_layers(traffic_dfs)
    test_dfs = construct_dict_2_layers(traffic_dfs)
    for dev, direction, dfs in iterate_2layer_dict(traffic_dfs):
        split_index = int(len(dfs) * (1 - test_size))
        train_dfs[dev][direction] = dfs.iloc[:split_index]
        test_dfs[dev][direction] = dfs.iloc[split_index:]

    return train_dfs, test_dfs
