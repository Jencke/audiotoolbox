import numpy as np
from scipy.stats import norm

def brickwall(signal, fs, low_f, high_f):
    '''Brickwall bandpass filter in frequency domain.

    Bandpass filters an input signal by setting all frequencies
    outside of the passband [low_f, high_f] to zero.

    Parameters:
    -----------
    signal : ndarray
        The input signal
    fs :  scalar
        The signals sampling rate in Hz
    low_f : scalar or None
        The lower cutoff frequency in Hz
    high_f : scalar
        The upper cutoff frequency in Hz

    Returns
    -------
    ndarray
        The filtered signal

    '''

    spec = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1. / fs)
    sel_freq = ~((np.abs(freqs) <= high_f) & (np.abs(freqs) >= low_f))
    spec[sel_freq] = 0
    filtered_signal = np.fft.ifft(spec)
    filtered_signal = np.real_if_close(filtered_signal, 1000)

    return filtered_signal


def gauss(signal, fs, low_f, high_f):
    '''Gauss bandpass filter in frequency domain.

    Bandpass filters an input signal by multiplying a gaussian
    function in the frequency domain. The cutoff frequencies are
    defined as the -3dB points

    Parameters:
    -----------
    signal : ndarray
        The input signal
    fs :  scalar
        The signals sampling rate in Hz
    low_f : scalar or None
        The lower cutoff frequency in Hz setting the lower frequency
        to None will center the the filter at zero and thus create a
        low-pass filter with a bandwidth of 0.5 * high_f
    high_f : scalar
        The upper cutoff frequency in Hz

    Returns
    -------
    ndarray
        The filtered signal

    '''
    spec = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1. / fs)

    if low_f == None:
        cf = 0
        bw = high_f * 2
        # half_width = high_f
    else:
        cf = (low_f + high_f) / 2
        bw = np.abs(high_f - low_f)

    # Calculate the std param to gain -3dB (amplitude) at the corner
    # frequencies
    db3val = np.sqrt(0.5)
    s = np.sqrt(-(bw / 2)**2 / (2 * np.log(db3val)))

    mag_spec1 = np.exp(-(freqs - cf)**2 / (2 * s**2))
    mag_spec1[freqs < 0] = 0
    mag_spec2 = np.exp(-(freqs + cf)**2 / (2 * s**2))
    mag_spec2[freqs >= 0] = 0
    # mag_spec = norm.pdf(freqs, loc=cf, scale=half_width)
    spec *= (mag_spec1 + mag_spec2)

    filtered_signal = np.fft.ifft(spec)
    filtered_signal = np.real_if_close(filtered_signal, 1000)

    return filtered_signal
