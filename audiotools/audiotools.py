'''
Some simple helper functions for dealing with audiosignals
'''

import numpy as np

def pad_for_fft(signal):
    '''Zero buffer a signal with zeros so that it reaches the next closest
       :math`$2^n$` length.

       This Function attaches zeros to a signal to adjust the length of the signal
       to a multiple of 2 for efficent FFT calculation.

       Parameters:
       -----------
       signal : ndarray
           The input signal

       Returns:
       --------
       ndarray : The zero bufferd output signal.

    '''
    exponent = np.ceil(np.log2(len(signal)))
    n_out = 2**exponent
    out_signal = np.zeros(int(n_out))
    out_signal[:len(signal)] = signal
    return out_signal


def zeropad(signal, number):
    '''Add a number of zeros to both sides of a signal.

        The input signal.
    number: int
        The number of zeros to add to the signal

    Returns:
    --------
    ndarray : The zero bufferd input signal

    '''
    zeros = np.zeros(number)
    signal_out = np.concatenate([zeros, signal, zeros])
    return signal_out


def cos_amp_modulator(signal, modulator_freq, fs, mod_index=1):
    '''Cosinus amplitude modulator

    Returns a cosinus amplitude modulator following the equation:
    ..math:: 1 + m * \cos{2 * \pi * f_m * t}

    where m is the modulation depth, f_m is the modualtion frequency
    and t is the

    Parameters:
    -----------
    signal : ndarray
        An input array that is used to determine the length of the
        modulator.

    modulator_freq : float
        The frequency of the cosine modulator.

    fs : float
        The sample frequency of the input signal.

    mod_index: float, optional
        The modulation index. (Default = 1)

    Returns:
    --------
    ndarray : The modulator

    '''

    if isinstance(signal, np.ndarray):
        time = get_time(signal, fs)
    elif isinstance(signal, int):
        time = get_time(np.zeros(signal), fs)
    else:
        raise TypeError("Signal must be numpy ndarray or int")

    modulator = 1 + mod_index * np.cos(2 * np.pi * modulator_freq * time)

    return modulator

def time2phase(time, frequency):
    '''Time to phase for a given frequency

    Parameters:
    -----------
    time : ndarray
        The time values to convert

    Returns:
    --------
    ndarray : converted phase values

    '''

    phase = time * frequency * (2 * np.pi)
    return phase

def phase2time(phase, frequency):
    '''Pase to Time for a given frequency

    Parameters:
    -----------
    phase : ndarray
        The phase values to convert

    Returns:
    --------
    ndarray : converted time values

    '''

    time = phase / (2 * np.pi) / frequency
    return time

def generate_tone(frequency, duration, fs, start_phase=0, endpoint=False):
    '''Sine tone with a given frequency, duration and sampling rate.

    This function will generate a pure tone with of a given duration
    at a given sampling rate. By default, the first sample will be
    evaluated at 0 and the duration will be the real stimulus duration.

    if endpoint is set to True, the duration will be considerd
    as the maxium point in time so that the function calculates one
    more sample. The stimulus duration is now duration + (1. / fs)

    Parameters:
    -----------
    frequency : scalar
        The tone frequency in Hz.
    duration : scalar
        The tone duration in seconds.
    fs : scalar
        The sampling rate for the tone.
    start_phase : scalar, optional
        The starting phase of the sine tone.
    endpoint : bool, optional
        Whether to generate an additional sample so that
        duration = time at last sample.

    Returns:
    --------
    ndarray : The sine tone

    '''
    len_signal = int(np.round(duration * fs))
    len_signal += 1 if endpoint else 0
    time = np.linspace(0, duration, len_signal, endpoint)
    tone = np.sin(2 * np.pi * frequency * time + start_phase)
    return tone

def get_time(signal, fs):
    '''Time axis of a given signal.

    This function generates a time axis for a given signal at a given
    sample rate.

    Parameters:
    -----------
    signal : ndarray
        The input signal for which to generate the time axis
    fs : scalar
        The sampling rate in Hz

    Returns:
    --------
    ndarray : The time axis in seconds

    '''
    dt = 1. / fs
    max_time = len(signal) * dt
    time = np.arange(0, max_time , dt)
    return time


def cosine_fade_window(signal, rise_time, fs):
    '''Cosine fade-in and fade-out window.

    This function generates a window function with a cosine fade in
    and fade out.

    Parameters:
    -----------
    signal: ndarray
        The length of the array will be used to determin the window length.
    rise_time : scalar
        Duration of the cosine fade in and fade out in seconds. The number of samples
        is determined via rounding to the nearest integer value.
    fs : scalar
        The sampling rate in Hz

    Returns:
    --------
    ndarray : The fading window

    '''
    r = int(np.round(rise_time * fs))
    window = np.ones(len(signal))
    flank = 0.5 * (1 + np.cos(np.pi / r * (np.arange(r) - r)))
    window[:r] = flank
    window[-r:] = flank[::-1]
    return window

def zero_buffer(signal, number):
    '''Add a number of zeros to both sides of a signal

    Parameters:
    -----------
    signal: ndarray
        The input Signal
    number : int
        The number of zeros that should be added

    Returns:
    --------
    ndarray : The bufferd signal

    '''
    assert isinstance(number, int)

    buf = np.zeros(number)
    signal = np.concatenate([buf, signal, buf])

    return signal


def delay_signal(signal, delay, fs):
    '''Delay by phase shifting in the frequncy domain.

    This function delays a given signal in the frequncy domain
    allowing for subsample time shifts.

    Parameters
    ----------
    signal : ndarray
        The signal to shift
    delay : scalar
        The delay in seconds
    fs :  scalar
        The signals sampling rate in Hz

    Returns
    -------
     ndarray : A array of shape [N, 2] where N is the length of the
         input signal. [:, 0] is the 0 padded original signal, [:, 1]
         the delayed signal

    '''

    #Only Positive Delays allowed
    assert delay >= 0

    # save the original length of the signal
    len_sig = len(signal)

    #due to the cyclic nature of the shift, pad the signal with
    #enough zeros
    n_pad = np.int(np.ceil(np.abs(delay * fs)))
    pad = np.zeros(n_pad)
    signal = np.concatenate([pad, signal, pad])

    #Apply FFT
    signal = pad_for_fft(signal)
    ft_signal = np.fft.fft(signal)

    #Calculate the phases need for shifting and apply them to the
    #spectrum
    freqs = np.fft.fftfreq(len(ft_signal), 1. / fs)
    ft_signal *= np.exp(-1j * 2 * np.pi * delay * freqs)

    #Inverse transform the spectrum and leave away the imag. part if
    #it is really small
    shifted_signal = np.fft.ifft(ft_signal)
    shifted_signal = np.real_if_close(shifted_signal, 1000)

    both = np.column_stack([signal, shifted_signal])

    # cut away the buffering
    both = both[n_pad:len_sig + 2 * n_pad, :]
    return both

def calc_dbspl(signal):
    '''Calculate the dB (SPL) value of a given signal.

    Parameters:
    -----------
    signal : ndarray
        The input signal

    Returns:
    --------
    float :
        The dB (SPL) value

    '''

    p0 = 20e-6
    rms_val = np.sqrt(np.mean(signal**2))
    dbspl_val = 20 * np.log10(rms_val / p0)

    return dbspl_val

def set_dbspl(signal, dbspl_val):
    '''Adjust signal amplitude to a given dbspl value.

    Parameters:
    -----------
    signal : ndarray
        The input signal
    dbspl_val : float
        The dbspl value to reach

    Returns:
    --------
    ndarray :
        The amplitude adjusted signal

    '''

    rms_val = np.sqrt(np.mean(signal**2))
    p0 = 20e-6 #ref_value

    factor = (p0 * 10**(float(dbspl_val) / 20)) / rms_val

    return signal * factor

def get_bark_limits():
    '''Limits of the Bark scale

    Returns the limit of the Bark scale as defined in [1]_.


    Returns
    -------
    list : Limits of the Bark scale

    References
    ----------
    .. [1] Zwicker, E. (1961). Subdivision of the audible frequency range into
           critical bands (frequenzgruppen). The Journal of the Acoustical
           Society of America, 33(2),
           248-248. http://dx.doi.org/10.1121/1.1908630

    '''
    bark_table = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                  1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                  4400, 5300, 6400, 7700, 9500, 12000, 15500]
    return bark_table

def freq_to_bark(frequency, use_table=False):
    '''Frequency to Bark conversion

    Converts a given sound frequency in Hz into the Bark scale using
    The equation by [2]_ or the original table by [1]_.

    Parameters
    ----------
    frequency: scalar or ndarray
        The frequency in Hz. Value has to be between 20 and 15500 Hz
    use_table: bool
        If True, the original table by [1]_ instead of the equation by [2]_
        is used. This also results in the CB beeing returned as integers.
        (Default = False)

    Returns
    -------
    scalar or ndarray : The Critical Bandwith in bark

    References
    ----------
    .. [1] Zwicker, E. (1961). Subdivision of the audible frequency range into
           critical bands (frequenzgruppen). The Journal of the Acoustical
           Society of America, 33(2),
           248-248. http://dx.doi.org/10.1121/1.19086f30
    .. [2] Traunmueller, H. (1990). Analytical expressions for the tonotopic
           sensory scale. The Journal of the Acoustical Society of America,
           88(1), 97-100. http://dx.doi.org/10.1121/1.399849

    '''
    assert np.all(frequency >= 20)
    assert np.all(frequency < 15500)

    if use_table:
        #Only use the table with no intermdiate values
        bark_table = np.array(get_bark_limits())
        scale_limits = zip(bark_table[:-1], bark_table[1:])
        i = 0
        cb_val = np.zeros(len(frequency))
        for lower, upper in scale_limits:
            in_border = (frequency >= lower) & (frequency < upper)
            cb_val[in_border] = i
            i += 1
        return cb_val
    else:
        cb_val = (26.81 * frequency / (1960 + frequency)) - 0.53
        if min(cb_val) < 2.0:
            cb_val[cb_val < 2.0] += 0.15 * (2 - cb_val[cb_val < 2.0])
        if max(cb_val) > 20.1:
            cb_val[cb_val > 20.1] += 0.22 * (cb_val[cb_val > 20.1] - 20.1)
        return cb_val
