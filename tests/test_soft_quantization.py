import numpy as np
import scipy

from features.gaussian_quantizer import GaussianQuantizer, multivariate_sampling_from_states
from features.packet_scaler import PacketScaler


def test_sampling(gaussian_quantizer):
    gmm = gaussian_quantizer.gmm_from
    sampled = multivariate_sampling_from_states(gmm, np.zeros(1000))
    expected_first_comp_means = gmm.means_[0]
    first_comp_means = np.mean(sampled, axis=0)
    assert np.isclose(first_comp_means, expected_first_comp_means, atol=1e-3).all()
    first_comp_std = np.std(sampled, axis=0)
    expected_first_comp_std = np.sqrt(np.diag(gmm.covariances_[0]))
    assert np.isclose(first_comp_std, expected_first_comp_std, atol=1e-3).all()


def test_scaler(raw_host_stats):
    scaler = PacketScaler()
    source_features = raw_host_stats.loc[:, ('PS', 'IAT, sec')].values
    scaled_features = scaler.transform(source_features.copy())
    assert np.isclose(source_features, scaler.inverse_transform(scaled_features), atol=10e-9).all()


def get_ks_stat(orig_values, gen_values):
    ks = scipy.stats.ks_2samp(orig_values, gen_values)
    return ks.statistic


def test_soft_quantizer(raw_host_stats):
    from sklearn.metrics import mean_absolute_error

    source_features = raw_host_stats.loc[:, ('PS', 'IAT, sec')].values
    directions = raw_host_stats['is_client'].values

    gaussian_quantizer = GaussianQuantizer().fit(source_features, directions, min_comp=5, max_comp=20)

    q_tokens = gaussian_quantizer.transform(source_features, directions, prepend_with_init_tokens=10)
    dec_features = gaussian_quantizer.inverse_transform(q_tokens, prob_sampling=False)

    mae = mean_absolute_error(source_features, dec_features)
    print(f'MAE: {mae}')
    assert mae < 0.39
    assert get_ks_stat(source_features[:, 0], dec_features[:, 0]) < 0.88
    assert get_ks_stat(source_features[:, 1], dec_features[:, 1]) < 0.45

    prob_dec_features = gaussian_quantizer.inverse_transform(q_tokens, prob_sampling=True)
    mae = mean_absolute_error(source_features, prob_dec_features)
    print(f'MAE with probabilistic sampling: {mae}')
    assert mae < 1.07
    assert get_ks_stat(source_features[:, 0], prob_dec_features[:, 0]) < 0.5
    assert get_ks_stat(source_features[:, 1], prob_dec_features[:, 1]) < 0.06
