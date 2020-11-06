from pprint import pprint

import numpy as np
import pandas as pd
import scipy

from pcap_parsing.parsed_fields import select_features


def packets_to_throughput(packets, resolution='1S'):
    # replace indexes with DateTime format
    df = pd.Series(
        packets[:, 0],
        index=pd.to_datetime(np.cumsum(packets[:, 1]), unit='ms')
    )
    throughput = df.resample(resolution).sum()
    return throughput.values


def get_ks_2sample_stat(orig_values, gen_values):
    return scipy.stats.ks_2samp(orig_values, gen_values).statistic


def evaluate_traffic(gen_df, test_df, verbose=True):
    restored_packets, from_idx = select_features(gen_df)
    client_gen_packets = restored_packets[from_idx]
    server_gen_packets = restored_packets[~from_idx]

    src_packets, src_from_id = select_features(test_df)
    client_src_packets = src_packets[src_from_id]
    server_src_packets = src_packets[~src_from_id]

    metrics = {
        'KS_2sample_PS_client': get_ks_2sample_stat(client_src_packets[:, 0], client_gen_packets[:, 0]),
        'KS_2sample_IAT_client': get_ks_2sample_stat(client_src_packets[:, 1], client_gen_packets[:, 1]),
        'KS_2sample_PS_server': get_ks_2sample_stat(server_src_packets[:, 0], server_gen_packets[:, 0]),
        'KS_2sample_IAT_server': get_ks_2sample_stat(server_src_packets[:, 1], server_gen_packets[:, 1]),
        'KS_2sample_thrpt_client': get_ks_2sample_stat(packets_to_throughput(client_src_packets),
                                                       packets_to_throughput(client_gen_packets)),
        'KS_2sample_thrpt_server': get_ks_2sample_stat(packets_to_throughput(server_src_packets),
                                                       packets_to_throughput(server_gen_packets)),
    }
    if verbose:
        pprint(metrics)
    return metrics
