import numpy as np

from features.auto_select_model_by_bic import auto_select_model_by_bic
from features.data_utils import load_train_test_dataset
from features.evaluation import evaluate_traffic
from hmm_generator.model import HMM, HMMGenerator
from pcap_parsing.parsed_fields import select_features


def test_bic(gmm_state):
    gmm_state = gmm_state.reshape(-1, 1)
    model, bic = auto_select_model_by_bic(HMM, gmm_state, min_comp=1, max_comp=5, return_bic_dict=True)
    assert list(map(lambda x: int(x), bic.values())) == [67655281, 65607561, 65370031, 64610293, 64498306]
    print(model.sample())


def test_sampling(raw_host_stats_path):
    train, test = load_train_test_dataset(raw_host_stats_path)
    model = HMMGenerator()

    model.fit(*select_features(train), min_comp=10, max_comp=12)
    gen = model.sample_packets_like(test)
    metrics = evaluate_traffic(gen, test)
    print(metrics)
    assert np.isclose(metrics['KS_2sample_thrpt_client'], 0.33, atol=1e-2)
    assert np.isclose(metrics['KS_2sample_thrpt_server'], 0.33, atol=1e-2)

