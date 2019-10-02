'''
Some simple helper functions for dealing with audiosignals
'''
from warnings import warn

import numpy as np
from numpy import pi

from scipy.interpolate import interp1d
from scipy.signal import hilbert
from scipy.signal.windows import hann

from .filter import brickwall

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

    if signal.ndim == 1:
        n_channels = 1
    else:
        n_channels = signal.shape[1]

    exponent = np.ceil(np.log2(len(signal)))
    n_out = 2**exponent
    if n_channels == 1:
        out_signal = np.zeros(int(n_out))
    else:
        out_signal = np.zeros([int(n_out), n_channels])
    out_signal[:len(signal)] = signal

    return out_signal

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

    modulator = 1 + mod_index * np.cos(2 * pi * modulator_freq * time)

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

    phase = time * frequency * (2 * pi)
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

    time = phase / (2 * pi) / frequency
    return time

def nsamples(duration, fs):
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

    Returns:
    --------
    int : number of samples in the signal

    '''
    len_signal = int(np.round(duration * fs))

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
        noise = brickwall(noise, fs, low_f, high_f)
    return noise

def generate_corr_noise(duration, fs, corr=0, cf=None, bw=None, seed=None):
    # generate two noise vectors
    noise_a = generate_noise(duration, fs, seed=seed)
    noise_b = generate_noise(duration, fs, seed=seed)

    # use Gram-Schmidt to generate orthogonal noise. This makes shure
    # that the two noise vectors are of equal power.
    Q, R = np.linalg.qr(np.column_stack([noise_a, noise_b]))
    noise_a = Q[:, 0] / np.abs(Q).max()
    noise_b = Q[:, 1] / np.abs(Q).max()

    alpha = corr
    beta = np.sqrt(1 - corr**2)

    # Generate partially corelated noise using the two channel method
    if corr > 0:
        noise_b = alpha * noise_a + beta * noise_b

    if cf:
        assert bw
        low_f = cf - 0.5 * bw
        high_f = cf + 0.5 * bw
        noise_a = brickwall(noise_a, fs, low_f, high_f)
        noise_b = brickwall(noise_b, fs, low_f, high_f)

    return noise_a, noise_b

def generate_tone(frequency, duration, fs, start_phase=0):
    '''Sine tone with a given frequency, duration and sampling rate.

    This function will generate a pure tone following the equation:
    .. math:: cos(2\pi f t + \phi_0)
    where f is the frequency, t is the time and phi_0 the starting phase.
    The first evulated timepoint is 0.

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

    Returns:
    --------
    ndarray : The sine tone

    '''
    nsamp = nsamples(duration, fs)
    time = get_time(nsamp, fs)
    tone = np.cos(2 * pi * frequency * time + start_phase)
    return tone

def get_time(signal, fs):
    '''Time axis of a given signal.

    This function generates a time axis for a given signal at a given
    sample rate.

    Parameters:
    -----------
    signal : ndarray or int
        The input signal for which to generate the time axis, or the
        number of samples for which to calculate the time axis

    fs : scalar
        The sampling rate in Hz

    Returns:
    --------
    ndarray : The time axis in seconds

    '''

    dt = 1. / fs

    if isinstance(signal, np.ndarray):
        nsamp = len(signal)
    elif isinstance(signal, int):
        nsamp = signal
    else:
        raise TypeError('Signal must be int or ndarray')

    max_time = nsamp * dt
    time = np.arange(0, max_time , dt)

    # Sometimes, due to numerics arange creates an extra sample which
    # needs to be removed
    if len(time) == nsamp + 1:
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

    r = nsamples(rise_time, fs)
    window = np.ones(len(signal) - 2 * n_zeros)
    flank = 0.5 * (1 + np.cos(pi / r * (np.arange(r) - r)))
    window[:r] = flank
    window[-r:] = flank[::-1]

    window = zeropad(window, n_zeros)

    # If the signal has multiple channels, extend the window to match
    # the shape
    if signal.ndim > 1:
        window = np.column_stack([window] * signal.shape[1])

    return window

def hann_fade_window(signal, rise_time, fs):
    '''Hann fade-in and fade-out window.

    This function generates a window function with a hann fade in
    and fade out.

    Parameters:
    -----------
    signal: ndarray
        The length of the array will be used to determin the window length.
    rise_time : scalar
        Duration of the hann fade in and fade out in seconds. The
        number of samples is determined via rounding to the nearest
        integer value.
    fs : scalar
        The sampling rate in Hz

    Returns:
    --------
    ndarray : The fading window

    '''

    n_samp = len(signal)
    window = np.ones(n_samp)
    n_rise = nsamples(rise_time, fs)
    flanks = hann(2 * n_rise)
    rise = flanks[:n_rise]
    decay = flanks[n_rise:]
    window[:n_rise] = rise
    window[-n_rise:] = decay

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
    r = int(np.round(rise_time * fs)) + 1 #number of values in window
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

def zeropad(signal, number):
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

    signal_out = np.concatenate([buf, signal, buf])

    return signal_out


def zero_buffer(signal, number):
    '''Depricated use zeropad instead

    '''

    warn("zero_buffer is deprecated, use zeropad instead",
         DeprecationWarning)

    signal = zeropad(signal, number)
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
    ft_signal *= np.exp(-1j * 2 * pi * delay * freqs)

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

    b_f = ((0.4 * 10**((l_dbspl + l_u) / 10 - 9))**alpha_f
           - (0.4 * 10**((t_f + l_u) / 10 -9))**alpha_f + 0.005135)
    l_phon = 40 * np.log10(b_f) + 94

    if limit:
        # Definition only valid starting from 20 phon
        assert l_phon >= 20

        if 20 <= frequency <= 4500:
            assert l_phon <= 90
        elif 4500 < frequency <= 12500:
            assert l_phon <= 80

    return l_phon

def calc_bandwidth(fc, scale='cbw'):
    '''Calculate approximation of auditory filter bandwidth

    This Function calculates aproximations for the auditory filter
    bandwidth using differnt concepts:

     - cbw: Use the critical bandwidth concept following [1]_
     - erb: Use the equivalent rectangular bandwith concept following [2]_

    Equation used for critical bandwidth:
    .. math:: B = 25 + 75 (1 + 1.4 \frac{f_c}{1000}^2)^0.69

    Equation used for critical equivalent rectangular bandwith:
    .. math:: B = 24.7 (4.37 \frac{f_c}{1000} + 1)

    Parameters:
    -----------
    fc : float or ndarray
      center frequency in Hz

    scale : str
      String indicating the scale that should be used possible values:
      'cbw' or 'erb'. (default='cbw')

        ..[1] Zwicker, E., & Terhardt, E. (1980). Analytical expressions for
              critical-band rate and critical bandwidth as a function of
              frequency. The Journal of the Acoustical Society of America,
              68(5), 1523-1525.

        ..[2] Glasberg, B. R., & Moore, B. C. (1990). Derivation of auditory
              filter shapes from notched-noise data. Hearing Research, 47(1-2),
              103-138.

    '''

    if 'cbw' in scale:
        bw = 25 + 75 * (1 + 1.4 * (fc / 1000.)**2)**0.69
    elif 'erb' in scale:
        bw = 24.7 * (4.37 * (fc / 1000.) + 1)

    return bw

def extract_binaural_differences(signal1, signal2, log_levels=True):
    '''Extract the binaural differences between two narrowband signals

    This function extimates the binaural evelope difference as well as the
    phase difference by applying the hilbert transform.

    The envelope difference is defined as the hilbert envelope of the
    first signal minus the hilbert envelope of the second signal while
    the phase difference is defined as the hilbert phase of the first
    minus the hilbert phase of the second.

    Due to the use of a hilbert transform, this approach should only be used on signals
    with a relatively narrow bandwidth.

    Parameters:
    -----------
    signal1 : ndarray
        The first input signal
    signal2 : ndarray
        The second input signal
    log_levels : bool, optional
        Defines whether the envelope difference is returned in db
        default = True

    Returns
    -------
    ipd : ndarray
        The phase difference
    env_diff : ndarray
        The envelope difference

    '''

    trans1 = hilbert(signal1)
    trans2 = hilbert(signal2)
    env1 = np.abs(trans1)
    env2 = np.abs(trans2)
    phase1 = np.angle(trans1)
    phase2 = np.angle(trans2)

    ipd = phase1 - phase2

    if log_levels:
        env_diff = 20*(np.log10(env1) - np.log10(env2))
    else:
        env_diff = env1 - env2

    # Phase wrap if phase difference larger then +- pi
    while np.abs(ipd).max() > pi:
        first_occ = np.where(np.abs(ipd) > pi)[0][0]
        sign = np.sign(ipd[first_occ])
        ipd[first_occ:] = -2 * pi * sign + ipd[first_occ:]

    # If the signal envelopes are close to zero the ipd should be
    # zero. this fixes some instabilities with the hilbert transform
    # that result in phase jumps when the two signals only differ from
    # zero due to numerics
    is_zero = np.isclose(env1, 0, atol=1e-6) & np.isclose(env2, 0, atol=1e-6)
    ipd[is_zero] = 0
    return ipd, env_diff

def schroeder_phase(harmonics, amplitudes, phi0=0.):
    '''Phases for a schroeder phase harmonic complex

    This function calculates the phases for a schroeder phase harmonic
    comlex following eq. 11 of [1]_


    Parameters:
    -----------
    harmonics : ndarray
        A vector of harmonics for which the schroeder phases should be
        calculated
    amplitudes : ndarray
        A vector of amplitudes for the given harmonics
    phi0 : scalar
        The starting phase of the first harmonic (default = 0)

    Returns:
    --------
    ndarray :
        The phase values for the harmonic compontents

    ..[1] Schroeder, M. (1970). Synthesis of low-peak-factor signals and
          binary sequences with low autocorrelation (corresp.). IEEE
          Transactions on Information Theory, 16(1),
          85–89.

    '''
    harmonics = np.array(harmonics)
    amplitudes = np.array(amplitudes)
    power = 0.5 * amplitudes**2
    power /= power.sum()

    phi_schroeder = np.zeros(len(harmonics))
    for i_n, n in enumerate(harmonics):
        phi_shift = 2 * pi * np.sum((n - harmonics[:i_n]) * power[:i_n])
        phi_schroeder[i_n] = phi0 - phi_shift

    return phi_schroeder

def crest_factor(signal, axis=0):
    '''Calculate crest factor

    Calculates the crest factor of the input signal. The crest factor
    is defined as:

    .. math:: C = \frac{|x_{peak}|}{x_{rms}}

    where :math:`x_{peak}` is the maximum of the absolute value and
    :math:`x{rms}` is the effective value of the signal.

    Parameters:
    -----------
    signal : ndarray
        The input signal
    axis : int
        The axis for which to calculate the crest factor (default = 0)

    Returns:
    --------
    scalar :
        The crest factor

    '''
    a_effective = np.sqrt(np.mean(signal**2, axis = axis))
    a_max = np.max(np.abs(signal), axis = axis)

    crest_factor = 20*np.log10(a_max / a_effective)

    return crest_factor

def phase_shift(signal, phase, fs):
    '''Shifts all frequency components of a signal by a constant phase.

    Shift all frequency components of a given signal by a constant
    phase by means of fFT transformation, phase shifting and inverse
    transformation.

    Parameters:
    -----------
    signal : ndarray
        The input signal
    phase : scalar
        The phase in rad by which the signal is shifted.

    Returns:
    --------
    ndarray :
        The phase shifted signal

    '''

    if signal.ndim == 1:
        n_channels = 1
    else:
        n_channels = signal.shape[1]

    n_signal = len(signal)
    signal = pad_for_fft(signal)
    i_signal = np.zeros([signal.shape[0], n_channels])

    for i in range(n_channels):
        if n_channels == 1:
            spec = np.fft.fft(signal)
        else:
            spec = np.fft.fft(signal[:, i])
        freqs = np.fft.fftfreq(len(signal), 1. / fs)

        shift_val = np.exp(1j * phase * np.sign(freqs))
        spec *= shift_val
        i_signal[:, i] = np.real_if_close(np.fft.ifft(spec), 3000)

    if n_channels == 1:
        ret = i_signal[:n_signal, 0]
    else:
        ret = i_signal[:n_signal, :]
    return  ret
