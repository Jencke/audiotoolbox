import numpy as np
from numpy import pi


def brickwall(signal, low_f, high_f, fs):
    '''Brickwall filter

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

    '''

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

# def brickwall_bandpass(signal, fc, bw, fs):
#     '''Brickwall bandpass filter

#     Bandpass filters an input signal by setting all frequency
#     outside of the passband defined by center frequency and bandwidth to zero.

#     Parameters
#     ----------
#     signal : ndarray
#         The input signal
#     fc : scalar
#         center frequency in Hz
#     bw : scalar
#         The bandwidth in Hz
#     fs :  scalar
#         The signals sampling rate in Hz

#     Returns
#     -------
#         The filtered signal

#     '''

#     low_f = fc - bw / 2
#     high_f = fc + bw / 2

#     filtered_signal = _brickwall(signal, low_f, high_f, fs)

#     return filtered_signal

# def brickwall_lowpass(signal, fc, fs):
#     '''Brickwall lowpass filter

#     Lowpass filters an input signal by setting all frequency
#     components above fc to zero.

#     Parameters
#     ----------
#     signal : ndarray
#         The input signal
#     fc : scalar
#         corner frequency in Hz
#     fs :  scalar
#         The signals sampling rate in Hz

#     Returns
#     -------
#         The filtered signal

#     '''

#     high_f = fc
#     low_f = 0

#     filtered_signal = _brickwall(signal, low_f, high_f, fs)

#     return filtered_signal

# def brickwall_highpass(signal, fc, fs):
#     '''Brickwall highpass filter

#     Highpass filters an input signal by setting all frequency
#     below fc to zero.

#     Parameters
#     ----------
#     signal : ndarray
#         The input signal
#     fc : scalar
#         corner frequency in Hz
#     fs :  scalar
#         The signals sampling rate in Hz

#     Returns
#     -------
#         The filtered signal

#     '''

#     filtered_signal = _brickwall(signal, fc, None, fs)

#     return filtered_signal
