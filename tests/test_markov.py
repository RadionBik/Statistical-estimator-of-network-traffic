import features.evaluation
from markov_baseline.model import MarkovSequenceGenerator


def test_markov_gen_transitions(gmm_state, gmm_model):

    generator = MarkovSequenceGenerator().fit(gmm_state)
    gen = generator.sample(len(gmm_state))
    stats = features.evaluation.calc_stats(gmm_state, gen)
    assert stats['distr_kl_div'] < 0.01
