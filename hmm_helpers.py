from traffic_helpers import *

from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

def print_hmm_gauss_model(model):
    print(f'Esimated Gaussian means:\n{pd.DataFrame(model.means_)}\n')
    print(f'Esimated transition matrix:\n{pd.DataFrame(model.transmat_)}\n')

def print_gmm(models, componentWeightThreshold=0.02):
    '''
    printEstimatedParametersEM() prints mixture components to STDOUT
    '''
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

def get_hmm_gaussian_models(traffic_dfs, n_comp=3, parameters=None):
    models = construct_dict_2_layers(traffic_dfs)
    for device, direction, df in iterate_dfs_plus(traffic_dfs):
        model = hmm.GaussianHMM(n_components=n_comp, covariance_type="full", n_iter=100, tol=0.001)
        print(f'Started fitting {device} in direction "{direction}"')
        if parameters:
            df = df[parameters]
        models[device][direction] = model.fit(df)    
    return models


def get_gmm(traffic_dfs, n_comp=None, parameters=None, w_conc_prior=0.01, w_conc_type='dirichlet_distribution'):
    models = construct_dict_2_layers(traffic_dfs)
    for device, direction, df in iterate_dfs_plus(traffic_dfs):
        lowest_bic = np.infty
        bic = []
        if not n_comp:
            comp_range = range(5,20)
            print(f'Started fitting {device} in direction "{direction}" with auto number of components')
        else:
            comp_range = range(n_comp,n_comp+1)
            print(f'Started fitting {device} in direction "{direction}"')

        for comp in comp_range:
            #model = BayesianGaussianMixture(n_components=comp,
            #                                covariance_type="full", 
            #                                max_iter=500, 
            #                                tol=0.001, 
            #                                weight_concentration_prior_type=w_conc_type, 
            #                                weight_concentration_prior=w_conc_prior)
            model = GaussianMixture(n_components=comp,
                                    covariance_type="full",)

            if parameters:
                df = df[parameters]
            model.fit(df)
                
            bic.append(model.bic(df))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                models[device][direction] = model    
    return models


def get_hmm_gmm(traffic_dfs, n_comp=3, n_mixture=3, parameters=None):
    models = construct_dict_2_layers(traffic_dfs)
    for device, direction, df in iterate_dfs_plus(traffic_dfs):
        model = hmm.GMMHMM(n_components=n_comp, n_mix=n_mixture, covariance_type="full", n_iter=200, tol=0.001)
        print(f'Started fitting {device} in direction "{direction}"')
        if parameters:
            df = df[parameters]
        models[device][direction] = model.fit(df)    
    return models

def get_kmeans(traffic_dfs, n_comp=3):
    clusters = construct_dict_2_layers(traffic_dfs)
    for device, direction, df in iterate_dfs_plus(traffic_dfs):
        clusters[device][direction] = KMeans(n_clusters=n_comp).fit_predict(df)
        #print(clusters[device][direction])
    return clusters


def gener_samples(models, scalers=None, sample_numb=500, truncate_negative=True, parameters=['IAT','pktLen']):

    '''
    generates sample features from GMM, HMM or GMM-HMM models. It is possible to set truncating 
    of negative values for plausibility of generated samples.
    '''
    samples = construct_dict_2_layers(models)
    states = construct_dict_2_layers(models)
    for device, direction, df in iterate_traffic_dict(models):
        model = models[device][direction]
        scaler = scalers[device][direction]
             
        if truncate_negative:
            fixed_samples = 0
            samples[device][direction] = pd.DataFrame(columns=parameters, index=[])
            states[device][direction] = np.array([])
            while fixed_samples!=sample_numb:
                samples_temp, states_temp = model.sample(sample_numb-fixed_samples)
                if scaler:
                    samples_temp = pd.DataFrame(scaler.inverse_transform(samples_temp), columns=parameters)

                positive_indexes = (samples_temp > 0).all(1)
                samples[device][direction] = samples[device][direction].append(samples_temp[positive_indexes])
                states[device][direction] = np.append(states[device][direction], states_temp[positive_indexes])
                
                #samples[device][direction].reset_index(inplace=True)
                fixed_samples = len(samples[device][direction])
                
        else:
            
            samples[device][direction], states[device][direction] = model.sample(sample_numb)
            if scaler:
                samples[device][direction] = pd.DataFrame(scaler.inverse_transform(samples[device][direction]), columns=parameters)
                #samples[device][direction][:,0] = samples[device][direction][:,0].astype(int) 

                
    return samples, states


def plot_hmm_transitions(models, samples, scalers=None, parameters=['pktLen','IAT'], print_model=False):
    '''
    TODO: fix labels and add plotting for model with defined features
    '''
    for device in samples:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[13, 6])
        for dir_numb, direction in enumerate(samples[device]):
            model = models[device][direction]
            if scalers:
                scaler = scalers[device][direction]
            if print_model:
                print_hmm_gauss_model(model)
            
            #plt.figure()
            # Plot the sampled data
            ax[dir_numb].plot(samples[device][direction][parameters[1]], samples[device][direction][parameters[0]],
                     '*', c='blue', alpha=0.1, markeredgecolor='none', label=f"direction '{direction}'")
                     #".", label=f"direction '{direction}'", ms=10, mfc="orange", alpha=0.7)

            # Indicate the component numbers
            means = model.means_
            if scalers:
                means = scaler.inverse_transform(model.means_)

            #for i, m in enumerate(means):
            #    ax[dir_numb].text(m[0], m[1], '%i'%(i+1),size=10, horizontalalignment='center',bbox=dict(alpha=.5, facecolor='w'))
            ax[dir_numb].legend(loc='best')

            ax[dir_numb].set_ylabel(parameters[0])
            ax[dir_numb].set_xlabel(parameters[1])
        #fig.show()

def normalize_by_rows(matrix):
    out_matrix = np.zeros(matrix.shape)
    row_len = matrix.shape[1]
    for i in range(matrix.shape[0]):
        denom = sum(matrix[i,:])
        if denom!=0:
            out_matrix[i,:] = matrix[i,:]/denom
        else:
            out_matrix[i,:] = np.full((1, row_len), 1/row_len)


    return out_matrix

def get_transition_matrix_with_training(state_numb, state_seq):
    found_states = list(set(state_seq))
    print(found_states)
    N = np.zeros((state_numb,state_numb))
    states = list(range(state_numb))
    for j, state_j in enumerate(states):
        for k, state_k in enumerate(states):
            #count number of each possible transition
            for t in range(len(state_seq)-1):
                if state_seq[t]==state_j and state_seq[t+1]==state_k:
                    N[j,k] += 1

    #normalize counts to probabilites, replacing on zeroed rows diagonal el-s with 1
    trans_matrix = normalize_by_rows(N)
    #show_transition_matrix(trans_matrix)
    return trans_matrix 

def get_hmm_from_gmm_pred_calc_transitions(gmm, pred):

    model = hmm.GaussianHMM(n_components=gmm.n_components, covariance_type="full")
    model.startprob_ = gmm.weights_
    model.means_ = gmm.means_
    model.covars_ = gmm.covariances_
    model.transmat_ = get_transition_matrix_with_training(gmm.n_components, pred)
    return model

def get_hmm_from_gmm_estimate_transitions(gmm, df):
    #combination of ‘s’ for startprob, ‘t’ for transmat, ‘m’ for means and ‘c’ for covars
    model = hmm.GaussianHMM(n_components=gmm.n_components, 
                            covariance_type="full",
                           params="t",
                           init_params="t")

    model.startprob_ = gmm.weights_
    model.means_ = gmm.means_
    model.covars_ = gmm.covariances_
    model.fit(df)
    
    return model