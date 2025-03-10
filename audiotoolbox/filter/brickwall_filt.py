"""Brickwall Filter."""

import numpy as np


def brickwall(signal, low_f, high_f, fs):
    """Brickwall filter.

    Bandpass filters an input signal by setting all frequency
    outside of the passband [low_f, high_f] to zero.

    Parameters
    ----------
    signal : ndarray
        The input signal
    low_f : scalar
        low cutoff in Hz
    high_f : scalar or None
        high cutoff in Hz, if None, high_f will be set to fs/2
    fs :  scalar
        The signals sampling rate in Hz

    Returns
    -------
        The filtered signal

    """
    if low_f is None and high_f is not None:
        low_f = 0
    elif low_f is not None and high_f is None:
        high_f = fs / 2
    elif low_f is None and high_f is None:
        raise Exception('low_f and/or high_f must be provided')

    spec = np.fft.fft(signal, axis=0)
    freqs = np.fft.fftfreq(len(signal), 1. / fs)
    sel_freq = ~((np.abs(freqs) <= high_f) & (np.abs(freqs) >= low_f))
    spec[sel_freq] = 0
    filtered_signal = np.fft.ifft(spec, axis=0)
    filtered_signal = np.real_if_close(filtered_signal, 1000)

    return filtered_signal
