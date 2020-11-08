from pprint import pprint

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

from pcap_parsing.parsed_fields import select_features, ParsedFields


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


def scatter_plot_with_kde(df, x='IAT, s', y='PS, bytes', fig_shifter=0, fig=None, title=''):
    if not fig:
        plt.figure(figsize=(13, 6))

    layout = (4, 8)

    x = pd.Series(df[x])
    y = pd.Series(df[y])

    kde_x = plt.subplot2grid(layout, (0, fig_shifter), colspan=3)
    kde_y = plt.subplot2grid(layout, (1, fig_shifter + 3), rowspan=3)
    scatter = plt.subplot2grid(layout, (1, fig_shifter), rowspan=3, colspan=3)

    sc_ax = sns.scatterplot(x=x, y=y, data=df, ax=scatter, alpha=0.5)
    if fig_shifter == 4:
        sc_ax.set_ylabel('')
    kde_x.set_xlim(sc_ax.get_xlim())
    kde_y.set_ylim(sc_ax.get_ylim())
    kde_x.axis('off')
    kde_y.axis('off')
    if title:
        kde_x.set_title(title)

    sns.distplot(x, vertical=0, ax=kde_x)
    sns.distplot(y, vertical=1, ax=kde_y)


def plot_packets_dist(stats, x=ParsedFields.iat, y=ParsedFields.ps):
    fig = plt.figure(figsize=(13, 6))

    from_idx = stats[ParsedFields.is_source]
    for idx, direction in enumerate(['От источника', 'К источнику']):
        if direction == 'От источника':
            packets = stats[from_idx]
        else:
            packets = stats[~from_idx]
        scatter_plot_with_kde(packets, x=x, y=y, fig=fig, fig_shifter=idx*4, title=direction)
    plt.tight_layout()


def calc_stats(original, generated, prefix='', verbose=True):
    stats = {}
    min_len = min(len(original), len(generated))
    original = original[:min_len]
    generated = generated[:min_len]
    stats['series_spearman'] = spearmanr(original, generated)[0]
    stats['series_pearson'] = pearsonr(original, generated)[0]
    stats['distr_ks2s'] = scipy.stats.ks_2samp(original, generated)[0]
    stats['distr_kl_div'] = get_KL_divergence_pdf(original, generated)
    stats['distr_wasserstein_dist'] = get_wasserstein_distance_pdf(original, generated)
    if prefix:
        stats = {f'{prefix}_{key}': value for key, value in stats.items()}
    if verbose:
        pprint(stats)
    return stats


def get_KL_divergence_pdf(orig_values, gen_values):
    x_values = np.linspace(0, max(orig_values), 100)
    kde_orig = scipy.stats.gaussian_kde(orig_values)(x_values)
    kde_gen = scipy.stats.gaussian_kde(gen_values)(x_values)
    return scipy.stats.entropy(kde_orig, kde_gen)


def get_wasserstein_distance_pdf(orig_values, gen_values):
    n_samples = 100
    x_values = np.linspace(0, max(orig_values), n_samples)
    kde_orig = scipy.stats.gaussian_kde(orig_values)(x_values)
    kde_gen = scipy.stats.gaussian_kde(gen_values)(x_values)
    kde_orig /= sum(kde_orig)
    kde_gen /= sum(kde_gen)

    return scipy.stats.wasserstein_distance(kde_orig * n_samples, kde_gen * n_samples)
