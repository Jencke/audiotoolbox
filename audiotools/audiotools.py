'''
Some simple helper functions for dealing with audiosignals
'''

import numpy as np
from scipy.interpolate import interp1d

COLOR_R = '#d65c5c'
COLOR_L = '#5c5cd6'

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

    signal : ndarray
        The input signal.
    number : int
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

def nsamples(duration, fs, endpoint=False):
    '''Calculates number of samples in a signal with a given duration.

    This function calculates the number of samples that will be
    returned when generating a signal with a certain duration and
    sampling rates.  The number is determined by multiplying the
    sampling rate with the duration and rounding to the next integer.

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
    int : number of samples in the signal

    '''
    len_signal = int(np.round(duration * fs))
    len_signal += 1 if endpoint else 0
    return len_signal

def generate_noise(duration, fs, cf=None, bw=None, seed=None):
    len_signal = nsamples(duration, fs)

    # Seed random number genarator
    np.random.seed(seed)
    noise = 2 * np.random.random(len_signal) - 1

    if cf:
        assert bw

        low_f = cf - 0.5 * bw
        high_f = cf + 0.5 * bw

        spec = np.fft.fft(noise)
        freqs = np.fft.fftfreq(len(noise), 1. / fs)
        sel_freq = ~((np.abs(freqs) <= high_f) & (np.abs(freqs) >= low_f))
        spec[sel_freq] = 0
        filtered_noise = np.fft.ifft(spec)
        filtered_noise = np.real_if_close(filtered_noise, 1000)
        noise = filtered_noise

    return noise


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
    len_signal = nsamples(duration, fs, endpoint)
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

    # Sometimes, due to numerics arange creates an extra sample which
    # needs to be removed
    if len(time) == len(signal) + 1:
        time = time[:-1]
    return time


def cosine_fade_window(signal, rise_time, fs, n_zeros=0):
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
    n_zeros : int, optional
        Number of zeros to add at the end and at the beginning of the window. (Default = 0)

    Returns:
    --------
    ndarray : The fading window

    '''

    assert isinstance(n_zeros, int)

    r = int(np.round(rise_time * fs))
    window = np.ones(len(signal) - 2 * n_zeros)
    flank = 0.5 * (1 + np.cos(np.pi / r * (np.arange(r) - r)))
    window[:r] = flank
    window[-r:] = flank[::-1]

    window = zero_buffer(window, n_zeros)

    if signal.ndim > 1:
        window = np.column_stack([window] * signal.shape[1])

    return window

def gaussian_fade_window(signal, rise_time, fs, cutoff=-60):
    '''Gausian fade-in and fade-out window.

    This function generates a window function with a gausian fade in
    and fade out. The gausian slope is cut at the level defined by the
    cutoff parameter

    Parameters:
    -----------
    signal: ndarray
        The length of the array will be used to determin the window length.
    rise_time : scalar
        Duration of the gaussian fade in and fade out in seconds. The
        value is measured from the cutof level until reaching a value
        of 1. The number of samples is determined via rounding to the
        nearest integer value.
    fs : scalar
        The sampling rate in Hz
    cutoff : scalar, optional
        The level at which the gausian slope is cut (default = -60dB)

    Returns:
    --------
    ndarray : The fading window

    '''
    cutoff_val = 10**(cutoff/ 20) # value at which to cut gaussian
    r = int(np.round(rise_time * fs)) #number of values in window
    window = np.ones(len(signal))
    win_time = np.linspace(0, rise_time, r)
    sigma = np.sqrt((-(rise_time)**2 / np.log(cutoff_val)) / 2)
    flank = np.exp(-(win_time - rise_time)**2 / (2 * sigma**2))

    # Set the beginning and and to the window to equal the flank
    window[:r-1] = flank[:-1]
    window[-r:] = flank[::-1]

    if signal.ndim > 1:
        window = np.column_stack([window] * signal.shape[1])

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

    if signal.ndim == 1:
        buf = np.zeros(number)
    else:
        buf = np.zeros([number, signal.shape[1]])

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
    if delay < 0:
        neg_delay = True
        delay = np.abs(delay)
    else:
        neg_delay = False

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

    # If negative delay then just invert the two signals
    if neg_delay:
        both = both[:,::-1]

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

def freqspace(min_frequency, max_frequency, n, scale='bark'):
    '''Calculate a given number of frequencies that eare equally spaced on the bark or erb scale.

    Returns n frequencies between min_frequency and max_frequency that are
    equally spaced on the bark or erb scale.

    Parameters
    ----------
    min_frequency: float
      minimal frequency in Hz

    max_frequency: float
      maximal frequency in Hz

    n: int
      Number of equally spaced frequencies

    scale: str
      scale to use 'bark' or 'erb'. (default='bark')

    Returns
    -------
    ndarray: n frequencies equally spaced in bark or erb
    '''

    if scale == 'bark':
        min_bark, max_bark = freq_to_bark(np.array([min_frequency, max_frequency]))
        barks = np.linspace(min_bark, max_bark, n)
        freqs = bark_to_freq(barks)
    elif scale == 'erb':
        min_erb, max_erb = freq_to_erb(np.array([min_frequency, max_frequency]))
        erbs = np.linspace(min_erb, max_erb, n)
        freqs = erb_to_freq(erbs)
    else:
        raise NotImplementedError('only ERB and Bark implemented')

    return freqs


def freqarange(min_frequency, max_frequency, step=1, scale='bark'):
    '''Calculate a of frequencies with a predifined spacing on the bark or erb scale.

    Returns frequencies between min_frequency and max_frequency with
    the stepsize step on the bark or erb scale.

    Parameters
    ----------
    min_frequency: float
      minimal frequency in Hz

    max_frequency: float
      maximal frequency in Hz

    step: float
      stepsize on the erb or bark scale

    scale: str
      scale to use 'bark' or 'erb'. (default='bark')

    Returns
    -------
    ndarray: frequencies spaced following step on bark or erb scale

    '''
    if scale == 'bark':
        min_bark, max_bark = freq_to_bark(np.array([min_frequency, max_frequency]))
        bark = np.arange(min_bark, max_bark, step)
        freqs = bark_to_freq(bark)
    elif scale == 'erb':
        min_erb, max_erb = freq_to_erb(np.array([min_frequency, max_frequency]))
        erbs = np.arange(min_erb, max_erb, step)
        freqs = erb_to_freq(erbs)
    else:
        raise NotImplementedError('only ERB and Bark implemented')

    return freqs

def bark_to_freq(bark):
    '''Bark to frequency conversion

    Converts a given value on the bark scale into frequency using the
    equation by [1]_

    Parameters
    ----------
    bark: scalar or ndarray
      The bark values

    Returns
    -------
    scalar or ndarray: The frequency in Hz

    References
    ----------
    ..[1] Traunmueller, H. (1990). Analytical expressions for the tonotopic
           sensory scale. The Journal of the Acoustical Society of America,
           88(1), 97-100. http://dx.doi.org/10.1121/1.399849
    '''

    #reverse apply corrections
    bark[bark < 2.0] = (bark[bark < 2.0] - 0.3) / 0.85
    bark[bark > 20.1] = (bark[bark > 20.1] + 4.422) / 1.22
    f = 1960 * (bark + 0.53) / (26.28 - bark)
    return f

def freq_to_bark(frequency, use_table=False):
    '''Frequency to Bark conversion

    Converts a given sound frequency in Hz into the Bark scale using
    The equation by [2]_ or the original table by [1]_.

    Parameters
    ----------
    frequency: scalar or ndarray
        The frequency in Hz. Value has to be between 20 and 15500 Hz
    use_table: bool, optional
        If True, the original table by [1]_ instead of the equation by [2]_
        is used. This also results in the CB beeing returned as integers.
        (default = False)

    Returns
    -------
    scalar or ndarray : The Critical Bandwith in bark

    References
    ----------
    ..[1] Zwicker, E. (1961). Subdivision of the audible frequency range into
           critical bands (frequenzgruppen). The Journal of the Acoustical
           Society of America, 33(2),
           248-248. http://dx.doi.org/10.1121/1.19086f30
    ..[2] Traunmueller, H. (1990). Analytical expressions for the tonotopic
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

def freq_to_erb(frequency):
    '''Frequency to number of ERBs conversion

    Calculates the number of erbs for a given sound frequency in Hz using the
    equation by [1]_

    Parameters
    ----------
    frequency: scalar or ndarray
        The frequency in Hz.

    Returns
    -------
    scalar or ndarray : The number of erbs corresponding to the frequency

    References
    ----------
    ..[2] Glasberg, B. R., & Moore, B. C. (1990). Derivation of auditory
          filter shapes from notched-noise data. Hearing Research, 47(1-2),
          103-138.
    '''

    n_erb = (1000. / (24.7 * 4.37)) * np.log(4.37 * frequency / 1000 + 1)
    return n_erb

def erb_to_freq(n_erb):
    '''number of ERBs to Frequency conversion

    Calculates the frequency from a given number of ERBs using
    equation by [1]_

    Parameters
    ----------
    n_erb: scalar or ndarray
        The number of ERBs

    Returns
    -------
    scalar or ndarray : The corresponding frequency

    References
    ----------
    ..[2] Glasberg, B. R., & Moore, B. C. (1990). Derivation of auditory
          filter shapes from notched-noise data. Hearing Research, 47(1-2),
          103-138.
    '''
    fkhz = (np.exp(n_erb * (24.7 * 4.37) / 1000) - 1) / 4.37
    return fkhz * 1000



def phon_to_dbspl(frequency, l_phon, interpolate=False, limit=True):
    '''Sound pressure levels from loudness level (following DIN ISO 226:2006-04)

    Calulates the sound pressure level at a given frequency that is necessary to
    reach a specific loudness level following DIN ISO 226:2006-04

    The normed values are tabulated for the following frequencies and sound pressure levels:
     1. 20phon to 90phon
       * 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
       * 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000
     2. 20phon to 80phon
       *5000, 6300, 8000, 10000, 12500

    Values for other frequencies can be interpolated (cubic spline) by setting the
    parameter interpolate to True. The check for correct sound pressure levels can be
    switched off by setting limit=False. In both cases, the results are not covered by
    the DIN ISO norm

    Parameters
    ----------
    frequency : scalar
        The frequency in Hz. must be one of the tabulated values above if interpolate = False
    l_phon : scalar
        loudness level that should be converted
    interpolate : bool, optional
        Defines whether the tabulated values from the norm should be interpolated.
        If set to True, the tabulated values will be interpolated using a cubic spline
        (default = False)
    limit : bool, optional
        Defines whether the limits of the norm should be checked (default = True)

    Returns
    -------
    scalar : The soundpressure level in dB SPL
    '''
    if limit:
        # Definition only valid starting from 20 phon
        assert l_phon >= 20

        if 20 <= frequency <= 4500:
            assert l_phon <= 90
        elif 4500 < frequency <= 12500:
            assert l_phon <= 80

    # Equation Parameters
    frequency_list = np.array([20, 25, 31.5,
                               40, 50, 63,
                               80, 100, 125,
                               160, 200, 250,
                               315, 400, 500,
                               630, 800, 1000,
                               1250, 1600, 2000,
                               2500, 3150, 4000,
                               5000, 6300, 8000,
                               10000, 12500])

    alpha_f_list = np.array([0.532, 0.506, 0.480,
                             0.455, 0.432, 0.409,
                             0.387, 0.367, 0.349,
                             0.330, 0.315, 0.301,
                             0.288, 0.276, 0.267,
                             0.259, 0.253, 0.250,
                             0.246, 0.244, 0.243,
                             0.243, 0.243, 0.242,
                             0.242, 0.245, 0.254,
                             0.271, 0.301])

    # transfer function normed at 1000Hz
    l_u_list = np.array([-31.6, -27.2, -23.0,
                         -19.1, -15.9, -13.0,
                         -10.3, -8.1, -6.2,
                         -4.5, -3.1, -2.0,
                         -1.1, -0.4, 0.0,
                         0.3, 0.5, 0.0,
                         -2.7, -4.1, -1.0,
                         1.7, 2.5, 1.2,
                         -2.1, -7.1, -11.2,
                         -10.7, -3.1])

    # Hearing threshold t_f
    t_f_list = np.array([78.5, 68.7, 59.5,
                         51.1, 44.0, 37.5,
                         31.5, 26.5, 22.1,
                         17.9, 14.4, 11.4,
                         8.6, 6.2, 4.4,
                         3.0, 2.2, 2.4,
                         3.5, 1.7, -1.3,
                         -4.2, -6.0, -5.4,
                         -1.5, 6.0, 12.6,
                         13.9, 12.3])

    if interpolate == False:
        assert frequency in frequency_list
        n_param = np.where(frequency_list == frequency)[0][0]

        alpha_f = alpha_f_list[n_param]
        l_u = l_u_list[n_param]
        t_f = t_f_list[n_param]
    else :
        i_type = 'cubic'
        alpha_f = interp1d(frequency_list, alpha_f_list, kind=i_type)(frequency)
        l_u = interp1d(frequency_list, l_u_list, kind=i_type)(frequency)
        t_f = interp1d(frequency_list, t_f_list, kind=i_type)(frequency)

    a_f = 4.47e-3 * (10**(0.025 * l_phon) - 1.15) + (0.4 * 10**((t_f + l_u) / 10 -9))**alpha_f
    l_pressure = 10 / alpha_f * np.log10(a_f) - l_u + 94

    return l_pressure

def dbspl_to_phon(frequency, l_dbspl, interpolate=False, limit=True):
    '''loudness levels from sound pressure level (following DIN ISO 226:2006-04)

    Calulates the loudness level at a given frequency from the sound
    pressure level following DIN ISO 226:2006-04

    The normed values are tabulated for the following frequencies and sound pressure levels:
     1. 20phon to 90phon
       * 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
       * 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000
     2. 20phon to 80phon
       *5000, 6300, 8000, 10000, 12500

    Values for other frequencies can be interpolated (cubic spline) by setting the
    parameter interpolate to True. The check for correct sound pressure levels can be
    switched off by setting limit=False. In poth cases, the results are not covered by
    the DIN ISO norm

    Parameters
    ----------
    frequency : scalar
        The frequency in Hz. must be one of the tabulated values above if interpolate = False
    l_dbspl : scalar
        sound pressure level that should be converted
    interpolate : bool, optional
        Defines whether the tabulated values from the norm should be interpolated.
        If set to True, the tabulated values will be interpolated using a cubic spline
        (default = False)
    limit : bool, optional
        Defines whether the limits of the norm should be checked (default = True)

    Returns
    -------
    scalar : The loudnes level level in dB SPL

    '''
    # Equation Parameters
    frequency_list = np.array([20, 25, 31.5,
                               40, 50, 63,
                               80, 100, 125,
                               160, 200, 250,
                               315, 400, 500,
                               630, 800, 1000,
                               1250, 1600, 2000,
                               2500, 3150, 4000,
                               5000, 6300, 8000,
                               10000, 12500])

    alpha_f_list = np.array([0.532, 0.506, 0.480,
                             0.455, 0.432, 0.409,
                             0.387, 0.367, 0.349,
                             0.330, 0.315, 0.301,
                             0.288, 0.276, 0.267,
                             0.259, 0.253, 0.250,
                             0.246, 0.244, 0.243,
                             0.243, 0.243, 0.242,
                             0.242, 0.245, 0.254,
                             0.271, 0.301])

    # transfer function normed at 1000Hz
    l_u_list = np.array([-31.6, -27.2, -23.0,
                         -19.1, -15.9, -13.0,
                         -10.3, -8.1, -6.2,
                         -4.5, -3.1, -2.0,
                         -1.1, -0.4, 0.0,
                         0.3, 0.5, 0.0,
                         -2.7, -4.1, -1.0,
                         1.7, 2.5, 1.2,
                         -2.1, -7.1, -11.2,
                         -10.7, -3.1])

    # Hearing threshold t_f
    t_f_list = np.array([78.5, 68.7, 59.5,
                         51.1, 44.0, 37.5,
                         31.5, 26.5, 22.1,
                         17.9, 14.4, 11.4,
                         8.6, 6.2, 4.4,
                         3.0, 2.2, 2.4,
                         3.5, 1.7, -1.3,
                         -4.2, -6.0, -5.4,
                         -1.5, 6.0, 12.6,
                         13.9, 12.3])

    if interpolate == False:
        assert frequency in frequency_list
        n_param = np.where(frequency_list == frequency)[0][0]

        alpha_f = alpha_f_list[n_param]
        l_u = l_u_list[n_param]
        t_f = t_f_list[n_param]
    else:
        i_type = 'cubic'
        alpha_f = interp1d(frequency_list, alpha_f_list, kind=i_type)(frequency)
        l_u = interp1d(frequency_list, l_u_list, kind=i_type)(frequency)
        t_f = interp1d(frequency_list, t_f_list, kind=i_type)(frequency)

    b_f = (0.4 * 10**((l_dbspl + l_u) / 10 - 9))**alpha_f - (0.4 * 10**((t_f + l_u) / 10 -9))**alpha_f + 0.005135
    l_phon = 40 * np.log10(b_f) + 94

    if limit:
        # Definition only valid starting from 20 phon
        assert l_phon >= 20

        if 20 <= frequency <= 4500:
            assert l_phon <= 90
        elif 4500 < frequency <= 12500:
            assert l_phon <= 80

    return l_phon
