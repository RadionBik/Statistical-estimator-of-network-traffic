import utils
import scipy
import pandas as pd
import numpy as np


def get_KL_divergence_pmf(orig_values, gen_values, state_numb=None):
    if not state_numb:
        state_numb = len(set(orig_values) & set(gen_values))

    pmf_orig = [len(orig_values[orig_values == state]) for state in range(state_numb)]
    pmf_gen = [len(gen_values[gen_values == state]) for state in range(state_numb)]

    return scipy.stats.entropy(pmf_orig, pmf_gen)


def get_KL_divergence_pdf(orig_values, gen_values, x_values):
    kde_orig = scipy.stats.gaussian_kde(orig_values)
    kde_gen = scipy.stats.gaussian_kde(gen_values)
    return scipy.stats.entropy(pd.Series(kde_orig(x_values)), pd.Series(kde_gen(x_values)))


def get_KL_divergence(orig_dfs, gen_dfs):
    '''
    calculates Kulback-Leibler divergence to describe similarity between distributions of 
    original and generated PktLens and IATs.
    '''

    KL_distances = utils.construct_dict_2_layers(orig_dfs)
    for device, direction, distance in utils.iterate_2layer_dict(KL_distances):
        min_len = min(len(orig_dfs[device][direction]), len(gen_dfs[device][direction]))
        KL_distances[device][direction] = {}

        parameters = list(orig_dfs[device][direction].columns)
        for parameter in parameters:
            orig_values = orig_dfs[device][direction][parameter].iloc[:min_len]
            gen_values = gen_dfs[device][direction][parameter].iloc[:min_len]
            x_values = np.linspace(0, max(orig_values), 100)
            distance = get_KL_divergence_pdf(orig_values, gen_values, x_values)

            KL_distances[device][direction].update({parameter: distance})

    print('Kulback-Leibler divergence:')
    for device in KL_distances:
        print(device)
        print(pd.DataFrame(KL_distances[device]))
    return KL_distances


def get_ks_2sample_test(orig_dfs, gen_dfs):
    '''
    tests whether PDFs can be treated as the same.  

    The small p-value is indicating that a test statistic as large as D would 
    be expected with probability p-value.
        
    If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis 
    that the distributions of the two samples are the same.

    '''
    ks_2s = utils.construct_dict_2_layers(orig_dfs)
    for device, direction, distance in utils.iterate_2layer_dict(ks_2s):
        min_len = min(len(orig_dfs[device][direction]), len(gen_dfs[device][direction]))
        ks_2s[device][direction] = {}

        parameters = list(orig_dfs[device][direction].columns)
        for parameter in parameters:
            orig_values = orig_dfs[device][direction][parameter].iloc[:min_len]
            gen_values = gen_dfs[device][direction][parameter].iloc[:min_len]

            ks_2s[device][direction][parameter] = {}
            ks = scipy.stats.ks_2samp(orig_values, gen_values)
            ks_df = pd.DataFrame([[ks.statistic, ks.pvalue]], columns=['statistic', 'p-value'],
                                 index=[direction + ' ,' + parameter])
            # rint('{}'.format(ks_df))
            ks_2s[device][direction][parameter].update({'p-value': ks.pvalue,
                                                        'statistic': ks.statistic})

    print('Kolmogorov-Smirnov 2-sample test:')
    for device, direction, ks in utils.iterate_2layer_dict(ks_2s):
        print(direction, device)
        print(pd.DataFrame(ks))

    return ks_2s


def get_percentiles_dfs(dfs, percentile_chunks=40):
    '''
    calculates feature percentiles (number of precentiles=percentile_chunks),
    aux function to describe the data, inspired by QQ plot.

    '''
    qq = utils.construct_dict_2_layers(dfs)
    for device, direction, df in utils.iterate_2layer_dict(dfs):
        qq[device][direction] = df.describe(percentiles=[x / percentile_chunks for x in range(percentile_chunks)]).iloc[
                                4:4 + percentile_chunks, :]
    return qq


def get_qq_r_comparison(orig_dfs, gener_dfs):
    '''
    Calcs Pearson correlattion coeff of percentile chunks of two traffic dfs. Antoher way to test 
    similarity of PDFs.

    https://stats.stackexchange.com/questions/27958/testing-randomly-generated-data-against-its-intended-distribution#27966

    '''
    print('Correlation of percentiles:')
    r_value = utils.construct_dict_2_layers(orig_dfs)

    qq_orig = get_percentiles_dfs(orig_dfs)
    qq_gen = get_percentiles_dfs(gener_dfs)

    for device, direction, df in utils.iterate_2layer_dict_copy(qq_orig):
        r_value[device][direction] = df.corrwith(qq_gen[device][direction])
        print('{}:\n{}'.format(direction, r_value[device][direction]))

    print('\n')
    return r_value


def calc_windowed_entropy_discrete(series, window=50):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    state_numb = len(set(series))
    windowed_entropy = []
    for i in range(0, len(series), window):
        series_slice = series.iloc[i:i + window]
        series_pmf = [len(series_slice[series_slice == state]) for state in range(state_numb)]
        windowed_entropy.append(scipy.stats.entropy(series_pmf))

    return pd.Series(windowed_entropy)


def calc_windowed_entropy_cont(series, window=50, kde_bins=1500):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    windowed_entropy = []
    for i in range(0, len(series), window):
        series_slice = series.iloc[i:i + window]
        if len(series_slice) < 10:
            continue

        series_pdf = scipy.stats.gaussian_kde(series_slice)
        # print(series_slice.values)
        x_values = np.linspace(0, max(series), kde_bins)
        entropy = scipy.stats.entropy(series_pdf(x_values))
        windowed_entropy.append(entropy)
    # print(windowed_entropy)

    return pd.Series(windowed_entropy).fillna(0)


def calc_smoothed_entropies_dfs(traffic_dfs, smoothing=5, window=50, kde_bins=500) -> pd.DataFrame:
    smoothed_entropies = {}

    for device, direction, parameter, value in utils.iterate_3layer_dict(traffic_dfs):
        entropy = calc_windowed_entropy_cont(value,
                                             kde_bins=kde_bins,
                                             window=window)

        legend = '{:4} | {:3} | avg={:1.2f}'.format(direction,
                                                    'PS' if parameter == 'pktLen' else parameter,
                                                    np.mean(entropy))

        smoothed_entropy = pd.Series(entropy).rolling(smoothing, center=True).mean()
        smoothed_entropies.update({legend: smoothed_entropy})

    return pd.DataFrame(smoothed_entropies)
