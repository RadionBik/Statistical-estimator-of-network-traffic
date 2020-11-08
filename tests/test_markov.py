from features.evaluation import calc_stats
from markov_baseline.model import MarkovSequenceGenerator
from markov_baseline import vomm


def test_markov_gen_transitions(gmm_state, gmm_model):
    generator = MarkovSequenceGenerator().fit(gmm_state)
    gen = generator.sample(len(gmm_state))
    stats = calc_stats(gmm_state, gen)
    assert stats['distr_kl_div'] < 0.01


def test_estim_vomm(gmm_state):
    model = vomm.ppm()
    model.fit(gmm_state, d=1)
    f_order_gen = model.generate_data(length=len(gmm_state))
    stats = calc_stats(gmm_state, f_order_gen)
    assert stats['distr_kl_div'] < 0.01
