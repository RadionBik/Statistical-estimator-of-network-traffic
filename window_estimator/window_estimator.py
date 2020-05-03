import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

'''
https://en.wikipedia.org/wiki/Pitch_detection_algorithm
https://gist.github.com/endolith/255291

здесь используется непараметрический подход к оценке доминантной частоты 
важным является оценка нижнего порога частота, т.к. именно нижняя доминанта определяет период временного сигнала
W. 
При этом, возможен случай когда трафик представялет собой некоррелированну последовательность, а значит
отсутствует какая-либо периодчиность в сигнале. В таком случае размер окна ограничавается сверху самой моделью 
согласно выражению (), до значения 512? -- взято из анализа последовательностей трафика


'''


def load_states(states_path):
    with open(states_path, 'r') as jsf:
        return np.array(json.load(jsf))


def calc_amplitude_spectrum(sequence):
    # norm='ortho' divides by sqrt(n)
    spectr = np.abs(np.fft.rfft(sequence - sequence.mean(), norm=None))
    freqs = np.fft.rfftfreq(sequence.size, 1)
    return spectr, freqs


def naive_window_estimation(spectr, freqs):
    top_one = spectr.argmax()
    return int(1 / freqs[top_one])


def plot_spectrum(ax, spectr, freqs, estimated_window=None, low_freq=None, high_freq=None):
    # f, ax = plt.subplots()
    ax.semilogx(freqs, spectr)
    ax.set_xlabel('Hz')
    ax.set_ylabel('Power spectrum')
    # max_spectr = max(spectr)
    leg_handles = []
    if estimated_window:
        est_freq = 1 / estimated_window
        line = ax.axvline(x=est_freq,
                          # ymin=0, ymax=max_spectr,
                          color='g', alpha=0.5, lw=2,
                          label='estimated F={:0.2} Hz'.format(est_freq))
        leg_handles.append(line)
        # ax.text(est_freq + 0.05 * est_freq, 0.8 * max_spectr, ' = {:0.2} Hz'.format(est_freq))

    if low_freq:
        line = ax.axvline(x=low_freq,
                          # ymin=0, ymax=max_spectr,
                          color='r', alpha=0.5, lw=2,
                          label='lower-limit F={:0.2} Hz'.format(low_freq))
        leg_handles.append(line)
        # ax.text(low_freq + 0.05 * low_freq, max_spectr, ' lim = {:0.2} Hz'.format(low_freq))
    if high_freq:
        line = ax.axvline(x=high_freq,
                          # ymin=0, ymax=max_spectr,
                          color='r', alpha=0.5, lw=2,
                          label='higher-limit F={:0.2} Hz'.format(high_freq))
        leg_handles.append(line)
        # ax.text(high_freq + 0.05 * high_freq, max_spectr, ' lim = {:0.2} Hz'.format(high_freq))
    if leg_handles:
        ax.legend(handles=leg_handles)
    return ax


def plot_acf(ax, sequence, lag, estimated_window=None):
    # f, ax = plt.subplots()
    pd.Series([pd.Series(sequence).autocorr(lag) for lag in range(lag)]).plot(grid=1, ax=ax)
    ax.set_xlabel('lag')
    ax.set_ylabel('ACF')
    if estimated_window:
        ax.axvline(x=estimated_window, ymin=0, ymax=1,
                   color='g', alpha=0.5, lw=2)
        ax.text(estimated_window + 0.02 * estimated_window, 0.7,
                f' = {estimated_window}')
    return ax


def filter_states(sequence, low_freq, high_freq=None):
    if high_freq:
        sos = signal.butter(50, Wn=(low_freq, high_freq), btype='bandpass', output='sos')
    else:
        # sos = signal.cheby2(50, rs=120, Wn=low_freq, btype='highpass', output='sos')
        sos = signal.butter(50, Wn=low_freq, btype='highpass', output='sos')
    filtered = signal.sosfilt(sos, sequence)
    return filtered


def freq_limited_window_estimation(spectrum_values, freqs, low_freq=None, high_freq=None):
    top_freq_index = spectrum_values.argmax()
    top_freq = freqs[top_freq_index]
    if low_freq and top_freq < low_freq:
        print('detected exceeding the lower limit frequency!')
        top_freq = low_freq
    elif high_freq and top_freq > high_freq:
        print('detected exceeding the higher limit frequency!')
        top_freq = high_freq
    return int(1 / top_freq)


def freq_bounded_window_estimation(spectrum_values, freqs, low_freq=None, high_freq=None) -> int:
    spectr_index_map = sorted({index: spectr for index, spectr in enumerate(spectrum_values)}.items(),
                              key=lambda x: x[1], reverse=True)

    print(spectrum_values, freqs)
    for m_index, m_spectr in spectr_index_map:
        if low_freq < freqs[m_index] < high_freq:
            break

    return int(1 / freqs[m_index])


def plot_report(states, axes=None, save_to=None, **filter_freqs):
    # spectr, freqs = calc_amplitude_spectrum(states)
    if axes is None:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(12, 4))

    # if filter_freqs:
    #     states = filter_states(states, **filter_freqs)
    freqs, spectr = signal.periodogram(states, scaling='spectrum')
    estimated_window = freq_bounded_window_estimation(spectr, freqs, **filter_freqs)
    plot_spectrum(axes[0], spectr, freqs, estimated_window, **filter_freqs)
    plot_acf(axes[1], states, estimated_window * 3, estimated_window)
    if save_to:
        plt.tight_layout()
        plt.savefig(save_to)
    return estimated_window


if __name__ == '__main__':
    for json_states in [
        'gmm_skype_to.json',
        'gmm_skype_from.json',
        'gmm_amazon_to.json',
        'gmm_amazon_from.json',
    ]:
        states = load_states(json_states)

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))

        low_freq = 1 / 512
        high_freq = None
        # states = filter_states(states, low_freq, high_freq)

        plot_report(states, axes, low_freq=low_freq, high_freq=high_freq)
        plt.show()
