#! /usr/bin/python3

'''
./compare_pcaps.py -o obj/skypeLANhome_UDP__192_168_0_102_18826__192_168_0_105_26454_1_components.pkl --pickle -g traffic_dumps/art_skypeLANhome_UDP__192_168_0_102_18826__192_168_0_105_26454_1_components.pcap -i flow


'''

from stat_estimator import *
from helper_functions import *
import scipy
import sklearn

def get_KL_divergence(original, generated, printOut=True):
    
    kl_divergence = defaultdict(dict)
    for dev_orig, dev_gen in zip(original, generated):
        for direction in original[dev_orig]:          
            kl_divergence[dev_gen][direction] = {'pktLen':{},'IAT':{}}
            for parameter in original[dev_orig][direction]:
                #print(type(original[dev_orig][direction][parameter]))
                orig_values = original[dev_orig][direction][parameter]
                gen_values = generated[dev_gen][direction][parameter]

                if isinstance(orig_values,sklearn.mixture.gaussian_mixture.GaussianMixture):
                    gmm_x = getXAxisValues(parameter, True)
                    #print(gmm_x)
                    orig_values = [ 0.00001 if x==0 else x for x in getGaussianMixtureValues(original[dev_orig][direction][parameter], gmm_x) ]
                
                if isinstance(gen_values,sklearn.mixture.gaussian_mixture.GaussianMixture):
                    gmm_x = getXAxisValues(parameter, True)
                    gen_values = [ 0.00001 if x==0 else x for x in getGaussianMixtureValues(generated[dev_gen][direction][parameter], gmm_x) ]
                    
                #make the compared lists equal in length
                length = min(len(orig_values), len(gen_values))
                kl_divergence[dev_gen][direction][parameter] = scipy.stats.entropy(orig_values[:length], gen_values[:length])
            
    if printOut:
        print_parameters('Kulbak-Leibler Divergence:',kl_divergence)
    return kl_divergence


def get_KS_2s_test(original, generated, printOut=True):
    
    ks_2s = defaultdict(dict)
    for dev_orig, dev_gen in zip(original, generated):
        for direction in original[dev_orig]:          
            ks_2s[dev_gen][direction] = {'pktLen':{},'IAT':{}}
            for parameter in original[dev_orig][direction]:
                if parameter=='ts':
                    continue
                orig_values = original[dev_orig][direction][parameter]
                gen_values = generated[dev_gen][direction][parameter]
                #print(type(orig_values))

                if isinstance(orig_values,sklearn.mixture.gaussian_mixture.GaussianMixture):
                    gmm_x = getXAxisValues(parameter, True)
                    #print(gmm_x)
                    orig_values = [ 0.00001 if x==0 else x for x in getGaussianMixtureValues(original[dev_orig][direction][parameter], gmm_x) ]
                
                if isinstance(gen_values,sklearn.mixture.gaussian_mixture.GaussianMixture):
                    gmm_x = getXAxisValues(parameter, True)
                    gen_values = [ 0.00001 if x==0 else x for x in getGaussianMixtureValues(generated[dev_gen][direction][parameter], gmm_x) ]
                    
                ks_2s[dev_gen][direction][parameter] = scipy.stats.ks_2samp(orig_values, gen_values)
            
    if printOut:
        print_parameters('Kolmogorov-Smirnov 2-sample test:',ks_2s)
    return ks_2s


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", help="original pcap to appoximate flow/device from or saved pickle of the estimations (if --pickle option is specified)",required=True)
    parser.add_argument("-g", help="pcap file with the generated traffic to appoximate flow/device from")    
    parser.add_argument("-i", help="specify identifier type to be read from the original file, either 'IP' (e.g. 172.16.0.1), or 'MAC' (e.g. xx:xx:xx:xx:xx:xx), or 'flow' (e.g. TCP 172.16.0.1:4444 172.16.0.2:8888))",default='flow')
    parser.add_argument("-f", help="file with devices or flow identifiers to process: MAC, IP, 5-tuple, e.g. 'TCP IPs:portS IPd:portD. if 'all' is specified, then every flow within the pcap will be estimated",default="addresses_to_check.txt")
    parser.add_argument("-percentiles", help="specify the lower and upper percentiles to remove possible outliers, e.g. 3,97. Default is 3,97", default='3,97')
    parser.add_argument("-n", help="estimate with N components. Opt for 'auto' if not sure, although takes more time to estimate", default='auto')
    parser.add_argument('--hist', dest='hist', action='store_true')
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--pickle', dest='pickle', action='store_true')
    parser.set_defaults(pickle=False, plot=False, hist=False)

    args = parser.parse_args()

    min_samples_to_estimate = 100
    percentiles = (int(args.percentiles.split(',')[0]), int(args.percentiles.split(',')[1]))

    generTraffic, generIdentifiers = getTrafficFeatures(args.g, args.f, args.i, percentiles,
                                                        min_samples_to_estimate=min_samples_to_estimate)

    generEstimations = getTrafficEstimations(generTraffic, args.n)[0]

    if not args.pickle:   
        origTraffic, origIdentifiers = getTrafficFeatures(args.o, args.f, args.i, percentiles,
                                                        min_samples_to_estimate=min_samples_to_estimate)
        origEstimations = getTrafficEstimations(origTraffic, args.n)[0]

    else:
        estimationsFileName = args.o.split('/')[len(args.o.split('/'))-1].split('.')[0]
        origEstimations = load_obj(estimationsFileName)[0]

    #printEstimatedParametersEM(origEstimations)
    #printEstimatedParametersEM(generEstimations)
    #plotHistograms(origTraffic)
    if args.hist:
        plot_hist(traffic=origTraffic, logScale=False)
        plot_hist(traffic=generTraffic, logScale=False)

    kl_divergence = get_KL_divergence(origEstimations, generEstimations)
    ks_2s_test =  get_KS_2s_test(origEstimations, generEstimations)
    
    if args.plot:
        plotEstimatedDistributions(generEstimations)
        plotEstimatedDistributions(origEstimations)

#https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
#https://www.thoughtco.com/definition-of-null-hypothesis-and-examples-605436
#https://www.graphpad.com/guides/prism/7/statistics/index.htm?interpreting_results_kolmogorov-smirnov_test.htm
#http://www.physics.csbsju.edu/stats/KS-test.html