import numpy as np

def brickwall(signal, fs, low_f, high_f):
    ''' Brickwall filter in frequency domain.

    '''
    spec = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1. / fs)
    sel_freq = ~((np.abs(freqs) <= high_f) & (np.abs(freqs) >= low_f))
    spec[sel_freq] = 0
    filtered_signal = np.fft.ifft(spec)
    filtered_signal = np.real_if_close(filtered_signal, 1000)
    signal = filtered_signal
    return filtered_signal
