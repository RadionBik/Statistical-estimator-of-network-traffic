import numpy as np
from traffic_helpers import *
import sklearn
import scipy
import random

def gauss_function(x, x0, sigma, amp=1):
    '''
    gauss_function() defines normalized gaussian function
    '''
    # return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))
    return amp * np.exp(-0.5 * ((x-x0)/sigma)**2) * 1/(sigma * np.sqrt(2*np.pi))



def get_EM_values_dict(estimatedParameters, x_values):

    generated_em_values = construct_new_dict(estimatedParameters)

    for device in estimatedParameters:
        for direction in estimatedParameters[device]:
            for parameter in estimatedParameters[device][direction]:
                if not isinstance(estimatedParameters[device][direction][parameter],\
                    sklearn.mixture.gaussian_mixture.GaussianMixture):
                    continue
                    # pdb.set_trace()
                generated_em_values[device][direction][parameter] = \
                    getGaussianMixtureValues(estimatedParameters[device][direction][parameter],
                                             x_values[device][direction][parameter])

    return generated_em_values


def getGaussianMixtureValues(gmm, gmm_x):
    # Construct function manually as sum of gaussians
    gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
    # pdb.set_trace()
    for w, m, var in zip(gmm.weights_.ravel(), gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())):
        gauss = gauss_function(x=gmm_x, x0=m, sigma=var)
        #print(np.trapz(gauss, gmm_x))
        normalizer = np.trapz(gauss, gmm_x)
        if normalizer > 0.000001:
            gmm_y_sum += gauss * w / normalizer
        else:
            gmm_y_sum += gauss * w

    return gmm_y_sum


def printGaussianMixtureParameters(gmm, componentWeightThreshold):
    for w, m, var in zip(gmm.weights_.ravel(), gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())):
        if w < componentWeightThreshold:
            continue
        print('W: {}, M: {}, V: {}'.format(w, m, var))


def printEstimatedParametersEM(estimationsEM, componentWeightThreshold=0.02):
    '''
    printEstimatedParametersEM() prints mixture components to STDOUT
    '''
    for device in estimationsEM:
        for direction in estimationsEM[device]:
            print('_______________________\n{1} {0}'.format(device, direction))
            for parameter in estimationsEM[device][direction]:
                if not isinstance(estimationsEM[device][direction][parameter],\
                                  sklearn.mixture.gaussian_mixture.GaussianMixture):
                    continue
                print("Parameter %s:" % parameter)
                gmm = estimationsEM[device][direction][parameter]
                for w, m, var in zip(gmm.weights_.ravel(), gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())):
                    if w < componentWeightThreshold:
                        continue
                    print(
                        'W: {0:1.3f}, M: {1:5.6f}, V: {2:5.8f}'.format(w, m, var))




def get_KDE_estimators_scipy(traffic):
    '''
    get_KDE_scipy() returns dict with estimated kernel density function,
    where it was estimated on data with values less than 95 percentile,
    in order to exclude outliers during bandwidth estimation process within 
    the scipy function
    '''

    print('Performing kernel density estimations...')
    kde_estimators = {}
    for device in traffic:
        kde_estimators[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:

                try:
                    kde_estimators[device][direction][parameter] = scipy.stats.gaussian_kde(
                    traffic[device][direction][parameter], bw_method='scott')
                except ValueError:
                    print('Could not esimate KDE for {} {} {}'.format(direction,device,parameter))
                    kde_estimators[device][direction][parameter] = traffic[device][direction][parameter]
                    continue

    return kde_estimators


def get_KDE_values(kde_estimators, x_values):

    kde_values = construct_new_dict(kde_estimators)
    for device in kde_estimators:
        for direction in kde_estimators[device]:
            for parameter in kde_estimators[device][direction]:
                estimator = kde_estimators[device][direction][parameter]
                #print(type(estimator))
                if isinstance(estimator, sklearn.neighbors.kde.KernelDensity):
                    kde_values[device][direction][parameter] = np.exp(estimator.score_samples(
                        np.array(x_values[device][direction][parameter]).reshape(-1,1)))
                elif isinstance(estimator, scipy.stats.kde.gaussian_kde):

                    kde_values[device][direction][parameter] = estimator.evaluate(
                        x_values[device][direction][parameter])
                else:
                    print('KDE: omitting {} {} {} for plotting'.format(direction, device, parameter))

    return kde_values


def get_EM_values_sklearn(estimatedParameters, sample_number=1000):

    generated_em_values = construct_new_dict(estimatedParameters)

    for device in estimatedParameters:
        for direction in estimatedParameters[device]:
            for parameter in estimatedParameters[device][direction]:
                generated_em_values[device][direction][parameter] = estimatedParameters[device][direction][parameter].sample(
                    sample_number)

    return generated_em_values


def get_EM_values_manual(estimatedParameters, trafficExtremeValues=[], packetNumber=1000):
    '''
    get_EM_values() generates packet properties given the EM-estimations,
    the max and min values of the parameters and the number of packets to generate
    '''

    generated_em_values = construct_new_dict(estimatedParameters)

    for device in estimatedParameters:
        for direction in estimatedParameters[device]:
            for parameter in estimatedParameters[device][direction]:

                mix = estimatedParameters[device][direction][parameter]
                if trafficExtremeValues:
                    paramLimit = trafficExtremeValues[device][direction][parameter]
                # generate parameters for the number of packets from the original dump
                for _ in range(packetNumber):
                    # select the component number that we are drawing sample from (Multinoulli)
                    component = np.random.choice(
                        np.arange(0, mix.n_components), p=mix.weights_)

                    # generate parameter until it fits the limits
                    suitable = False
                    while not suitable:
                        if parameter == 'pktLen':
                            # packet length in bytes must be integer
                            genParam = round(np.asscalar(random.gauss(
                                mix.means_[component], np.sqrt(mix.covariances_[component]))))
                        else:
                            genParam = np.asscalar(random.gauss(
                                mix.means_[component], np.sqrt(mix.covariances_[component])))

                        if not trafficExtremeValues or ((genParam <= paramLimit['max']) and (genParam >= paramLimit['min'])):
                            suitable = True

                    generated_em_values[device][direction][parameter].append(
                        genParam)

                # replace the first value of IAT with 0 for consistency
                if parameter == 'IAT':
                    generated_em_values[device][direction][parameter][0] = 0

    return generated_em_values
