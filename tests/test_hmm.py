from hmm_generator.model import HMM
from features.auto_select_model_by_bic import auto_select_model_by_bic


def test_bic(gmm_state):
    gmm_state = gmm_state.reshape(-1, 1)
    model, bic = auto_select_model_by_bic(HMM, gmm_state, min_comp=1, max_comp=5, return_bic_dict=True)
    assert list(map(lambda x: int(x), bic.values())) == [67655281, 65607561, 65370031, 64610293, 64498306]
