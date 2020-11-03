from hmmlearn import hmm

import markov_models
from markov_baseline.model import MarkovSequenceGenerator

import stat_metrics


# def test_estim_vomm(gmm_state):
#     import vomm
#     my_model = vomm.ppm()
#     my_model.fit(gmm_state, d=1)
#
#     f_order_gen = my_model.generate_data(length=len(gmm_state))
#     # import plotting
#     # plotting.plot_states(f_order_gen)
#     # plotting.plot_states(gmm_state)
#
#     stats = stat_metrics.calc_stats(gmm_state, f_order_gen)
#     assert stats['distr_kl_div'] < 0.02


def test_calc_transitions(gmm_state, gmm_model):
    n_comp = max(gmm_state) + 1
    model = hmm.GaussianHMM(n_components=n_comp, covariance_type="full")

    # doesn't affect the generated states
    model.startprob_ = gmm_model.weights_
    model.means_ = gmm_model.means_
    model.covars_ = gmm_model.covariances_

    model.transmat_ = markov_models.get_transition_matrix_with_training(n_comp, gmm_state)
    gen = model.sample(len(gmm_state))[1]
    stats = stat_metrics.calc_stats(gmm_state, gen)
    assert stats['distr_kl_div'] < 0.01


def test_markov_gen_transitions(gmm_state, gmm_model):

    generator = MarkovSequenceGenerator().fit(gmm_state)
    gen = generator.sample(len(gmm_state))
    stats = stat_metrics.calc_stats(gmm_state, gen)
    assert stats['distr_kl_div'] < 0.01
