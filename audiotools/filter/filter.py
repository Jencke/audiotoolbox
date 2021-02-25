import numpy as np
from scipy.stats import norm
from numpy import pi
from . import gammatone_filt as gt

def gammatone(signal, fc, bw, fs, order=4, attenuation_db='erb', return_complex=True):
    """Apply a gammatone filter to the signal

    Applys a gammatone filter following [1]_ to the input signal
    and returns the filtered signal.

    Parameters
    ----------
    signal : ndarray
        The input signal
    fs : int
      The sample frequency in Hz
    fc : scalar
      The center frequency of the filter in Hz
    bw : scalar
      The bandwidth of the filter in Hz
    order : int
      The filter order (default = 4)
    attenuation_db: scalar or 'erb'
      The attenuation at half bandwidth in dB (default = -3).
      If set to 'erb', the bandwidth is interpreted as the rectangular
      equivalent bw
    return_complex : bool
      Whether the complex filter output or only it's real
      part is returned (default = True)

    Returns
    -------
      The filtered signal.

    References
    ----------
    .. [1] Hohmann, V., Frequency analysis and synthesis using a
          Gammatone filterbank, Acta Acustica, Vol 88 (2002), 43 -3442

    """

    b, a = gt.design_gammatone(fc, bw, fs, order, attenuation_db)

    out_signal = np.zeros_like(signal, complex)

    if signal.ndim > 1:
        n_channel = signal.shape[1]
        for i_c in range(n_channel):
            out_signal[:, i_c], _ = gt.gammatonefos_apply(signal[:, i_c], b, a, order)
    else:
        out_signal[:], _ = gt.gammatonefos_apply(signal, b, a, order)

    if not return_complex:
        out_signal = out_signal.real

    return out_signal


def brickwall_bandpass(signal, fc, bw, fs):
    '''Brickwall bandpass filter

    Bandpass filters an input signal by setting all frequency
    outside of the passband [low_f, high_f] to zero.

    Parameters
    ----------
    signal : ndarray
        The input signal
    fc : scalar
        center frequency in Hz
    high_f : scalar
        The upper cutoff frequency in Hz
    fs :  scalar
        The signals sampling rate in Hz

    Returns
    -------
        The filtered signal

    '''

    low_f = fc - bw / 2
    high_f = fc + bw / 2

    spec = np.fft.fft(signal, axis=0)
    freqs = np.fft.fftfreq(len(signal), 1. / fs)
    sel_freq = ~((np.abs(freqs) <= high_f) & (np.abs(freqs) >= low_f))
    spec[sel_freq] = 0
    filtered_signal = np.fft.ifft(spec, axis=0)
    filtered_signal = np.real_if_close(filtered_signal, 1000)

    return filtered_signal


# def middle_ear_filter(signal, fs):
#     f1 = 4000 / fs
#     f2 = 1000 / fs
#     q = 2 - np.cos(2 * pi * f1) - np.sqrt((np.cos(2 * pi * f1)-2)**2-1)
#     r = 2 - np.cos(2 * pi * f2) - np.sqrt((np.cos(2 * pi * f2)-2)**2-1)

#     sig_len = len(signal) + 2
#     n_channels = signal.shape[1]
#     y = np.zeros((sig_len, n_channels))
#     x = np.zeros((sig_len, n_channels))
#     x[2:] = signal

#     for i in range(2, sig_len):
#         y[i] = (+ (1 - q) * r * x[i]
#                 - (1 - q) * r * x[i - 1]
#                 - (q + r) * y[i - 1]
#                 - (q * r) * y[i - 2])
#     return y[2:]


# import matplotlib.pyplot as plt

# signal = np.random.random([1000000, 2])
# signal -= 0.5
# out = middle_ear_filter(signal, 100e3)

# plt.plot(np.abs(np.fft.fft(signal[:, 0])))

# plt.plot(np.abs(np.fft.fft(out[:, 0])) / np.abs(np.fft.fft(signal[:, 0])))
