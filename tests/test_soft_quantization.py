import numpy as np
from sklearn.metrics import mean_absolute_error

from features.evaluation import get_ks_2sample_stat
from features.gaussian_quantizer import GaussianQuantizer, multivariate_sampling_from_states
from features.packet_scaler import PacketScaler
from pcap_parsing.parsed_fields import select_features


def test_sampling(gaussian_quantizer):
    gmm = gaussian_quantizer.model_from
    sampled = multivariate_sampling_from_states(gmm, np.zeros(1000))
    expected_first_comp_means = gmm.means_[0]
    first_comp_means = np.mean(sampled, axis=0)
    assert np.isclose(first_comp_means, expected_first_comp_means, atol=1e-3).all()
    first_comp_std = np.std(sampled, axis=0)
    expected_first_comp_std = np.sqrt(np.diag(gmm.covariances_[0]))
    assert np.isclose(first_comp_std, expected_first_comp_std, atol=1e-3).all()


def test_scaler(raw_host_stats):
    scaler = PacketScaler()
    source_features = select_features(raw_host_stats)[0]
    scaled_features = scaler.transform(source_features.copy())
    reversed_features = scaler.inverse_transform(scaled_features)
    assert np.isclose(source_features, reversed_features, atol=10e-9).all()


def test_soft_quantizer(raw_host_stats):

    source_features, directions = select_features(raw_host_stats)
    gaussian_quantizer = GaussianQuantizer().fit(source_features, directions, min_comp=5, max_comp=20)

    q_tokens = gaussian_quantizer.transform(source_features, directions, prepend_with_init_tokens=10)
    dec_features, dec_directions = gaussian_quantizer.inverse_transform(q_tokens, prob_sampling=False)

    assert (directions == dec_directions).all()
    mean_exp_values = source_features.mean(axis=0)
    mape = mean_absolute_error(source_features / mean_exp_values, dec_features / mean_exp_values)
    print(f'MAPE: {mape}')
    assert mape < 0.035
    assert get_ks_2sample_stat(source_features[:, 0], dec_features[:, 0]) < 0.54
    assert get_ks_2sample_stat(source_features[:, 1], dec_features[:, 1]) < 0.73

    prob_dec_features, prob_dec_directions = gaussian_quantizer.inverse_transform(q_tokens, prob_sampling=True)
    mape = mean_absolute_error(source_features / mean_exp_values, prob_dec_features / mean_exp_values)
    print(f'MAPE with probabilistic sampling: {mape}')
    assert mape < 0.05
    assert get_ks_2sample_stat(source_features[:, 0], prob_dec_features[:, 0]) < 0.5
    assert get_ks_2sample_stat(source_features[:, 1], prob_dec_features[:, 1]) < 0.28
