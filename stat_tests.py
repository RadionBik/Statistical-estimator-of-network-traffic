from traffic_helpers import *
from plot_helpers import getXAxisValues
import scipy
import pandas as pd

def get_KL_divergence_value(orig_values, gen_values, x_values):

    kde_orig = scipy.stats.gaussian_kde(orig_values)
    kde_gen = scipy.stats.gaussian_kde(gen_values)
    return scipy.stats.entropy(pd.Series(kde_orig(x_values)), pd.Series(kde_gen(x_values)))


def get_KL_divergence(orig_dfs, gen_dfs):
    '''
    calculates Kulback-Leibler divergence to describe similarity between distributions of 
    original and generated PktLens and IATs.
    '''

    print('Kulback-Leibler divergence:')
    KL_distances = construct_dict_2_layers(orig_dfs)
    for device, direction, distance in iterate_traffic_dict(KL_distances):
        min_len = min(len(orig_dfs[device][direction]), len(gen_dfs[device][direction]))
        KL_distances[device][direction] = {}
        
        parameters = list(orig_dfs[device][direction].columns)
        for parameter in parameters:

            orig_values = orig_dfs[device][direction][parameter].iloc[:min_len]
            gen_values = gen_dfs[device][direction][parameter].iloc[:min_len]
            x_values = getXAxisValues(parameter, traffic=orig_values)
            distance = get_KL_divergence_value(orig_values, gen_values, x_values)

            KL_distances[device][direction].update({parameter : distance})
            print('{:4s}, {:7s}\t{:0.4f}'.format(direction,parameter, distance))
        
    return KL_distances

def get_ks_2sample_test(orig_dfs, gen_dfs):
    
    '''
    tests whether PDFs can be treated as the same.  

    The small p-value is indicating that a test statistic as large as D would 
    be expected with probability p-value.
        
    If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis 
    that the distributions of the two samples are the same.

    '''
    print('Kolmogorov-Smirnov 2-sample test:')
    ks_2s = construct_dict_2_layers(orig_dfs)
    for device, direction, distance in iterate_traffic_dict(ks_2s):
        min_len = min(len(orig_dfs[device][direction]), len(gen_dfs[device][direction]))
        ks_2s[device][direction] = {}
        
        parameters = list(orig_dfs[device][direction].columns)
        for parameter in parameters:
            orig_values = orig_dfs[device][direction][parameter].iloc[:min_len]
            gen_values = gen_dfs[device][direction][parameter].iloc[:min_len]
            
            ks_2s[device][direction][parameter] = {}
            ks = scipy.stats.ks_2samp(orig_values, gen_values)
            ks_df = pd.DataFrame([[ks.statistic, ks.pvalue]], columns=['statistic','p-value'], index=[direction+' ,'+parameter])
            print('{}'.format(ks_df))
            ks_2s[device][direction][parameter].update({'p-value': ks.pvalue, 
                                                        'statistic': ks.statistic})
    return ks_2s

def get_percentiles_dfs(dfs, percentile_chunks=40):
    '''
    calculates feature percentiles (number of precentiles=percentile_chunks),
    aux function to describe the data, inspired by QQ plot.

    '''
    qq = construct_dict_2_layers(dfs)
    for device, direction, df in iterate_traffic_dict(dfs):
        qq[device][direction] = df.describe(percentiles=[x/percentile_chunks for x in range(percentile_chunks)]).iloc[4:4+percentile_chunks,:]
    return qq

def get_qq_r_comparison(orig_dfs, gener_dfs):

    '''
    Calcs Pearson correlattion coeff of percentile chunks of two traffic dfs. Antoher way to test 
    similarity of PDFs.

    https://stats.stackexchange.com/questions/27958/testing-randomly-generated-data-against-its-intended-distribution#27966

    '''
    print('Correlation of percentiles:')
    r_value = construct_dict_2_layers(orig_dfs)

    qq_orig = get_percentiles_dfs(orig_dfs)
    qq_gen = get_percentiles_dfs(gener_dfs)

    for device, direction, df in iterate_dfs_plus(qq_orig):
        r_value[device][direction] = df.corrwith(qq_gen[device][direction])
        print('{}:\n{}'.format(direction, r_value[device][direction]))

    print('\n')
    return r_value