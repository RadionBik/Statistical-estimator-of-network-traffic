import logging
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn import mixture
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import utils

logger = logging.getLogger(__name__)


def gauss_function(x, x0, sigma, amp=1):
    """
    gauss_function() defines normalized gaussian function
    """
    # return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))
    return amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2) * 1 / (sigma * np.sqrt(2 * np.pi))


def get_EM_values_dict(estimatedParameters, x_values):
    generated_em_values = utils.construct_new_dict(estimatedParameters)

    for device in estimatedParameters:
        for direction in estimatedParameters[device]:
            for parameter in estimatedParameters[device][direction]:
                if not isinstance(estimatedParameters[device][direction][parameter],
                                  sklearn.mixture.GaussianMixture):
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
        # print(np.trapz(gauss, gmm_x))
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
    """
    printEstimatedParametersEM() prints mixture components to STDOUT
    """
    for device in estimationsEM:
        for direction in estimationsEM[device]:
            print('_______________________\n{1} {0}'.format(device, direction))
            for parameter in estimationsEM[device][direction]:
                if not isinstance(estimationsEM[device][direction][parameter],
                                  sklearn.mixture.GaussianMixture):
                    continue
                print("Parameter %s:" % parameter)
                gmm = estimationsEM[device][direction][parameter]
                for w, m, var in zip(gmm.weights_.ravel(), gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())):
                    if w < componentWeightThreshold:
                        continue
                    print(
                        'W: {0:1.3f}, M: {1:5.6f}, V: {2:5.8f}'.format(w, m, var))


def get_KDE_estimators_scipy(traffic):
    """
    get_KDE_scipy() returns dict with estimated kernel density function,
    where it was estimated on data with values less than 95 percentile,
    in order to exclude outliers during bandwidth estimation process within 
    the scipy function
    """

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
                    print('Could not esimate KDE for {} {} {}'.format(direction, device, parameter))
                    kde_estimators[device][direction][parameter] = traffic[device][direction][parameter]
                    continue

    return kde_estimators


def get_KDE_values(kde_estimators, x_values):
    kde_values = utils.construct_new_dict(kde_estimators)
    for device in kde_estimators:
        for direction in kde_estimators[device]:
            for parameter in kde_estimators[device][direction]:
                estimator = kde_estimators[device][direction][parameter]
                # print(type(estimator))
                if isinstance(estimator, sklearn.neighbors.KernelDensity):
                    kde_values[device][direction][parameter] = np.exp(estimator.score_samples(
                        np.array(x_values[device][direction][parameter]).reshape(-1, 1)))
                elif isinstance(estimator, scipy.stats.kde.gaussian_kde):

                    kde_values[device][direction][parameter] = estimator.validate(
                        x_values[device][direction][parameter])
                else:
                    print('KDE: omitting {} {} {} for plotting'.format(direction, device, parameter))

    return kde_values


def get_EM_values_sklearn(estimatedParameters, sample_number=1000):
    generated_em_values = utils.construct_new_dict(estimatedParameters)

    for device in estimatedParameters:
        for direction in estimatedParameters[device]:
            for parameter in estimatedParameters[device][direction]:
                generated_em_values[device][direction][parameter] = estimatedParameters[device][direction][
                    parameter].sample(
                    sample_number)

    return generated_em_values


def print_gmm(models, componentWeightThreshold=0.02):
    """
    printEstimatedParametersEM() prints mixture components to STDOUT
    """
    for device in models:
        for direction in models[device]:
            print('_______________________\n{1} {0}'.format(device, direction))
            if not isinstance(models[device][direction], BayesianGaussianMixture):
                continue
            gmm = models[device][direction]
            for w, m, var in zip(gmm.weights_.ravel(), gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())):
                if w < componentWeightThreshold:
                    continue
                print(
                    'W: {0:1.3f}, M: {1:5.6f}, V: {2:5.8f}'.format(w, m, var))


@utils.unpack_2layer_traffic_dict
def get_gmm(df,
            n_comp=None,
            parameters=None,
            sort_components=False,
            **kwargs):

    if not n_comp:
        comp_range = range(5, 20)
        logger.info(f'Started fitting GMM with auto number of components')
    else:
        comp_range = range(n_comp, n_comp + 1)
        logger.info(f'Started fitting GMM')

    models = []
    for comp in comp_range:
        if kwargs:
            model = BayesianGaussianMixture(n_components=comp,
                                            covariance_type="full",
                                            max_iter=500,
                                            tol=0.001,
                                            random_state=88,
                                            **kwargs)
        else:
            model = GaussianMixture(n_components=comp,
                                    covariance_type="full",
                                    random_state=88)

        df = df[parameters] if parameters else df

        model.fit(df)
        models.append(model)

    best_model = min(models, key=lambda x: x.bic(df))
    logger.info('Best BIC is with {} components'.format(best_model.n_components))

    if sort_components:
        sorted_indexes = np.linalg.norm(best_model.means_, axis=1).argsort()
        for attr in ['means_', 'covariances_', 'weights_']:
            setattr(best_model, attr, getattr(best_model, attr)[sorted_indexes])
        logger.info('reassigned components numbers according to their sorted norm')

    return best_model


# @utils.unpack_2layer_traffic_dict
def generate_features_from_gmm_states(gmm_model, states, scaler=None):
    gen_samples = np.zeros((len(states), gmm_model.means_.shape[1]))
    gen_packet = np.zeros(gmm_model.means_.shape[1])
    for i, state in enumerate(states.astype('int32')):
        positive = False
        while not positive:
            for feature in range(gmm_model.means_.shape[1]):
                mean = gmm_model.means_[state][feature]
                var = np.sqrt(gmm_model.covariances_[state][feature, feature])
                gen_packet[feature] = random.gauss(mean, var)
            if scaler:
                gen_packet = scaler.inverse_transform(gen_packet.reshape(1, -1))[0, :]
            if gen_packet[0] > 0 and gen_packet[1] > 0:
                positive = True
                gen_samples[i, :] = gen_packet

    return pd.DataFrame(gen_samples, columns=['IAT', 'pktLen'])


def estimateParametersEM(traffic, componentNumb=5):
    """
    estimateParametersEM() estimates statistical properties (Gauusian Mixtures)
    via EM-algorithm for IAT and pktLen parameters for each device/flow and returns
    dict with estimated mixture objects

    """
    gmm_estimates = {}
    for device in traffic:
        gmm_estimates[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:
                # logger.info('Estimating {}, direction: {}, parameter {}'.format(device,direction,parameter))
                deviceData = np.array(traffic[device][direction][parameter]).reshape(-1, 1)
                gmm_estimates[device][direction][parameter] = \
                    mixture.GaussianMixture(n_components=componentNumb, covariance_type='full', random_state=88)
                gmm_estimates[device][direction][parameter].fit(deviceData)
                # logger.info(gmm_estimates[device][direction][parameter].means_)
    return gmm_estimates


def estimateParametersBEM(traffic, componentNumb=5):
    """
    estimateParametersEM() estimates statistical properties (Gauusian Mixtures)
    via EM-algorithm for IAT and pktLen parameters for each device/flow and returns
    dict with estimated mixture objects

    """
    gmm_estimates = {}
    for device in traffic:
        gmm_estimates[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:
                # logger.info('Estimating {}, direction: {}, parameter {}'.format(device,direction,parameter))
                deviceData = np.array(traffic[device][direction][parameter]).reshape(-1, 1)
                gmm_estimates[device][direction][parameter] = \
                    mixture.BayesianGaussianMixture(n_components=componentNumb, covariance_type='full', random_state=88,
                                                    weight_concentration_prior=0.01)
                gmm_estimates[device][direction][parameter].fit(deviceData)
                # logger.info(gmm_estimates[device][direction][parameter].means_)
    return gmm_estimates


def estimate_parameters_EM_BIC(traffic, min_samples_to_estimate=15):
    logger.info('Estimating mixtures with EM-algorithm...')
    gmm_estimates = {}
    for device in traffic:
        gmm_estimates[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:

                if parameter == 'pktLen':
                    compensation = 0
                else:
                    compensation = 1

                if len(traffic[device][direction][parameter]) < min_samples_to_estimate:
                    logger.info('Could not apply EM for {} {} {}'.format(direction, device, parameter))
                    gmm_estimates[device][direction][parameter] = traffic[device][direction][parameter]
                    continue

                # set regularization to depend on max values, for nicier plots and estimation
                reg_cov = 10 ** (-round(1 / max(traffic[device][direction][parameter])) - compensation)
                deviceData = np.array(traffic[device][direction][parameter]).reshape(-1, 1)
                lowest_bic = np.infty
                bic = []
                for comp in range(1, 6):
                    gmm = mixture.GaussianMixture(n_components=comp, covariance_type='full', random_state=88,
                                                  reg_covar=reg_cov)
                    try:
                        gmm.fit(deviceData)
                        bic.append(gmm.bic(deviceData))
                        if bic[-1] < lowest_bic:
                            lowest_bic = bic[-1]
                            gmm_estimates[device][direction][parameter] = gmm
                    except ValueError:
                        logger.info('Not enough samples for {}. Stopped at {} components'.format(device, comp - 1))
                        break
                logger.info(
                    '{} {} {}: selected mixture with {} components ({} max)'.format(direction, device, parameter,
                                                                                    np.argmin(bic) + 1, len(bic)))

    return gmm_estimates


def get_best_KDE(traffic):
    kde_estimators = {}
    for device in traffic:
        kde_estimators[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:
                deviceData = np.array(traffic[device][direction][parameter]).reshape(-1, 1)
                # use grid search cross-validation to optimize the bandwidth
                params = {'bandwidth': np.logspace(-5, 1, 7)}
                grid = GridSearchCV(KernelDensity(), params)
                grid.fit(deviceData)

                logger.info("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

                kde_estimators[device][direction][parameter] = grid.best_estimator_

    return kde_estimators


def get_KDE(traffic):
    kde_estimators = {}
    for device in traffic:
        kde_estimators[device] = defaultdict(dict)
        for direction in traffic[device]:
            for parameter in ['IAT', 'pktLen']:
                if parameter == 'IAT':
                    bandwidthDef = 0.0002
                else:
                    bandwidthDef = 1

                deviceData = np.array(traffic[device][direction][parameter]).reshape(-1, 1)
                kde_estimators[device][direction][parameter] = KernelDensity(bandwidth=bandwidthDef).fit(deviceData)

    return kde_estimators


def get_traffic_estimations(extracted_traffic, component_numb, min_samples_to_estimate=15):
    if component_numb == 'auto':
        estimated_parameter_em = estimate_parameters_EM_BIC(extracted_traffic, min_samples_to_estimate)
    else:
        try:
            estimated_parameter_em = estimateParametersEM(extracted_traffic, int(component_numb))
        except ValueError:
            raise ValueError('wrong argument of component number. See help: -h')

    kde_estimators = get_KDE_estimators_scipy(extracted_traffic)
    # kde_estimators = get_best_KDE(extractedTraffic)

    logger.info('Finished estimating the traffic')

    return estimated_parameter_em, kde_estimators


@utils.unpack_2layer_traffic_dict
def get_mixture_state_predictions(mixture_model, traffic_df):
    return mixture_model.predict(traffic_df)