import numpy as np

import stat_metrics
from window_estimator import window_estimator


def test_kl_pmf():
    x1 = (np.random.random(1000)*10).astype(np.int)
    x2 = (np.random.random(1000) * 10).astype(np.int)
    res_pmf = stat_metrics.get_KL_divergence_pmf(x1, x1)
    assert np.isclose(res_pmf, 0)

    res_pmf = stat_metrics.get_KL_divergence_pmf(x1, x2)

    assert np.isclose(res_pmf, 0, atol=0.05)
    res_pdf = stat_metrics.get_KL_divergence_pdf(x1, x1)
    assert np.isclose(res_pdf, 0)

    res_pdf = stat_metrics.get_KL_divergence_pdf(x1, x2)
    assert np.isclose(res_pdf, 0, atol=0.05)


def test_window_est():
    spectrum_val = np.array([0.8, 0.4, 0.6, 0.1, 0.9])
    freqs = np.array([.1, .12, .2, .25, .3])
    low_freq = .11
    high_freq = .28
    wdw = window_estimator.freq_bounded_window_estimation(spectrum_val, freqs, low_freq=low_freq, high_freq=high_freq)
    assert wdw == int(1 / freqs[2])
