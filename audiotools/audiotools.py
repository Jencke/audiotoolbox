"""Function based interface to audiotools."""

import numpy as np
from numpy import pi
from scipy.interpolate import interp1d
from scipy.signal import hilbert

from .oaudio import Signal
from . import filter

COLOR_R = '#d65c5c'
COLOR_L = '#5c5cd6'


def _copy_to_dim(array, dim):
    if np.ndim(dim) == 0:
        dim = (dim,)

    # tile by the number of dimensions
    tiled_array = np.tile(array, (*dim[::-1], 1)).T
    # squeeze to remove axis of lenght 1
    tiled_array = np.squeeze(tiled_array)

    return tiled_array


def _duration_is_signal(duration, fs=None, n_channels=None):
    r"""Check if the duration which was passed was really a signal class."""
    inval = duration
    if isinstance(duration, Signal):
        real_duration = inval.duration
        real_fs = inval.fs
        real_nch = inval.n_channels
    elif isinstance(duration, np.ndarray):
        n_samples = len(duration)
        if fs is not None:
            real_duration = n_samples / fs
        else:
            real_duration = None
        real_fs = fs
        if np.ndim(duration) > 1:
            real_nch = duration.shape[1:]
        else:
            real_nch = 1
    else:
        real_duration = duration
        real_fs = fs
        real_nch = n_channels

    # assert not (real_fs is None)

    return real_duration, real_fs, real_nch


def from_wav(filename, fullscale=True):
    """Read signal from wav file"""
    from .oaudio import Signal
    from .wav import readwav

    wv, fs = readwav(filename, fullscale)

    if wv.ndim > 1:
        n_channels = wv.shape[1]
    else:
        n_channels = 1

    duration = wv.shape[0] / fs
    sig = Signal(n_channels, duration, fs)
    sig[:] = wv

    return sig


def pad_for_fft(signal):
    r"""Zero buffer a signal with zeros so that it reaches the next closest :math`$2^n$` length.

       This Function attaches zeros to a signal to adjust the length
       of the signal to a multiple of 2 for efficent FFT calculation.

       Parameters
       -----------
       signal : ndarray
           The input signal

       Returns
       --------
       ndarray : The zero bufferd output signal.

    """

    if signal.ndim == 1:
        n_channels = 1
    else:
        n_channels = signal.shape[1]

    n_out = nextpower2(len(signal))
    if n_channels == 1:
        out_signal = np.zeros(int(n_out))
    else:
        out_signal = np.zeros([int(n_out), n_channels])
    out_signal[:len(signal)] = signal

    return out_signal


def nextpower2(num):
    exponent = np.ceil(np.log2(num))
    n_out = int(2**exponent)
    return n_out


def band2rms(bandlevel, bw):
    r"""Convert bandlevel to rms level

    Assuming a white spectrum, this functions converts a Bandlevel in
    dB/sqrt(Hz) into the corresponding RMS levle in dB

    ..math:: L_{rms} = L_{band} + 10 \log_10(f_\delta)

    where :math:`f_\delta` is the bandwidth of the signal
    """

    rmslevel = bandlevel + 10 * np.log10(bw)

    return rmslevel


def rms2band(rmslevel, bw):
    r"""Convert bandlevel to rms level

    Assuming a white spectrum, this functions converts a rms level in db into
    into the corresponding bandlevel

    """

    bandlevel = rmslevel - 10 * np.log10(bw)

    return bandlevel


def cos_amp_modulator(duration, modulator_freq, fs=None, mod_index=1,
                      start_phase=0):
    r"""Cosinus amplitude modulator.

    Returns a cosinus amplitude modulator following the equation:

    ..  math:: 1 + m \cos{2 \pi f_m t \phi_{0}}

    where :math:`m` is the modulation depth, :math:`f_m` is the
    modualtion frequency and :math:`t` is the time.  :math;`\phi_0` is
    the start phase

    Parameters
    ----------
    duration : ndarray An input array that is used to determine the
    length of the modulator.

    modulator_freq : float The frequency of the cosine modulator.

    fs : float The sample frequency of the input signal.

    mod_index: float, optional The modulation index.  (Default = 1)

    Returns
    -------
    ndarray : The modulator

    See Also
    --------

    audiotools.Signal.add_cos_modulator
    """
    duration, fs, n_channels = _duration_is_signal(duration, fs)
    n_samples = nsamples(duration, fs)
    time = get_time(duration, fs)

    modulator = 1 + mod_index * np.cos(2 * pi * modulator_freq * time
                                       + start_phase)

    modulator = _copy_to_dim(modulator, n_channels)

    return modulator


def time2phase(time, frequency):
    r"""Time to phase for a given frequency.

    .. math:: \phi = 2 \pi t f

    Parameters
    -----------
    time : ndarray
        The time values to convert

    Returns
    --------
    converted phase values : ndarray

    """

    phase = time * frequency * (2 * pi)
    return phase


def phase2time(phase, frequency):
    r"""Pase to Time for a given frequency

    .. math:: t = \frac{\phi}{2 \pi f}

    Parameters
    -----------
    phase : ndarray
        The phase values to convert

    Returns
    --------
    converted time values : ndarray

    """

    time = phase / (2 * pi) / frequency
    return time


def nsamples(duration, fs=None):
    r"""Number of samples in a signal with a given duration.

    This function calculates the number of samples that will be
    returned when generating a signal with a certain duration and
    sampling rates.  The number is determined by multiplying the
    sampling rate with the duration and rounding to the next integer.

    Parameters
    -----------
    duration : scalar
        The signals duration in seconds. Or Signal class
    fs : scalar (optional)
        The sampling rate for the tone. Is ignored when Signal class is passed
    Returns
    --------
    number of samples in the signal : int

    """
    duration, fs, n_ch = _duration_is_signal(duration, fs)

    len_signal = int(np.round(duration * fs))

    return len_signal


def generate_low_noise_noise(duration, fc, bw, fs=None,
                             n_channels=1, n_rep=10, seed=None):
    r"""Low-noise Noise

    Generate Low-noise noise as defined in [1]_.

    Parameters
    -----------
    duration : scalar
        Noise duration in seconds
    fs : int
        Sampling frequency
    low_f : float
        Lower cut-off frequency
    high_f : float
        Higher cut-off frequency.
    n_rep : int
        Number of low-noise noise iterations (default=10)
    seed :
        seed for the random number generator.

    References
    ----------

    .. [1] Kohlrausch, A., Fassel, R., van der Heijden, M., Kortekaas,
           R., van de Par, S., Oxenham, A.J. and Püschel, D.,
           1997. Detection of tones in low-noise noise: Further
           evidence for the role of envelope fluctuations. Acta
           Acustica united with Acustica, 83(4), pp.659-669.

    """

    # Todo - need to fix duration
    duration, fs, n_ch = _duration_is_signal(duration, fs, n_channels)

    # Generate initial noise
    noise = generate_noise(duration, fs, ntype='white', n_channels=n_ch)
    noise = filter.brickwall(noise, fc - bw / 2, fc + bw / 2, fs)
    std = noise.std(axis=0)

    for i in range(n_rep):
        hilb = hilbert(noise, axis=0)
        env = abs(hilb)

        # diveide through envelope and restrict
        noise /= env
        noise = filter.brickwall(noise, fc - bw/2, fc + bw/2, fs)
        noise /= noise.std(axis=0) * std

    return noise


def generate_noise(duration, fs=None, ntype='white', n_channels=1, seed=None):
    r"""Generate Noise

    Generate gaussian noise with a variance of 1 and different
    spectral shapes. The noise is generated in the frequency domain
    using the gaussian pseudorandom generator ``numpy.random.randn``.
    The real and imaginary part of each frequency component is set
    using the psudorandom generator. Each frequency bin is then
    weighted dependent on the spectral shape. The resulting spektrum
    is then transformed into the time domain using ``numpy.fft.ifft``

    Weighting functions

     - white: :math:`w(f) = 1`
     - pink: :math:`w(f) = \frac{1}{\sqrt{f}}`
     - brown: :math:`w(f) = \frac{1}{f}`

    Parameters
    ----------
    duration : scalar or Signal
        Noise duration in seconds. If Signal object is passed, then
        duration is taken from object.
    fs : int (optional)
        Sampling frequency, If Signal object is passed, then
        duration is taken from object.
    ntype : {'white', 'pink', 'brown'}
        spectral shape of the noise
    n_channels : int (optional)
        number of indipendant noise channels. If Signal object is passed, then
        duration is taken from object.
    seed : int or 1-d array_like, optional
        Seed for `RandomState`.
        Must be convertible to 32 bit unsigned integers.

    Returns
    -------
    ndarray
        noise vector of the shape (NxM) where N is the number of samples
        and M >the number of channels

    See Also
    --------
    audiotools.Signal.add_noise

    """
    np.random.seed(seed)

    # If signal class is passed, get parameters directy from it
    inval = duration
    if isinstance(duration, Signal):
        duration = inval.duration
        fs = inval.fs
        n_channels = inval.n_channels

    len_signal = nsamples(duration, fs)

    # If noise type is white just use the random number generator
    if ntype == 'white':
        noise = np.random.randn(len_signal)
        noise -= noise.mean(axis=0)
        # normalize variance
        noise /= noise.std(axis=0)

        noise = _copy_to_dim(noise, n_channels)

        return np.squeeze(noise)

    # Otherwise create spectrum
    # Calculate length and number of fft samples
    len_signal = nsamples(duration, fs)
    nfft = nextpower2(len_signal)

    df = fs/nfft                # Frequency resolution
    nybin = nfft // 2 + 1       # nyquist bin

    lowbin = 1               # no offset start at one
    highbin = nybin

    freqs = np.arange(0, nybin) * df

    # amplitude weighting factor
    f_weights = np.zeros(nfft)
    if ntype == 'pink':
        # Power proportinal to 1 / f
        f_weights[lowbin:highbin] = 1. / np.sqrt(freqs[lowbin:])
    elif ntype == 'brown':
        # Power proportional to 1 / f**2
        f_weights[lowbin:highbin] = 1. / freqs[lowbin:]

    # generate noise
    a = np.zeros([nfft])
    b = np.zeros([nfft])
    a[lowbin:highbin] = np.random.randn(highbin - lowbin)
    b[lowbin:highbin] = np.random.randn(highbin - lowbin)
    fspec = a + 1j * b

    # Frequency weighting
    fspec *= f_weights

    noise = np.fft.ifft(fspec, axis=0)
    noise = np.real(noise)

    #  clip and remove offset due to clipping
    noise = noise[:len_signal]
    noise -= noise.mean()

    # Normalize the signal by its rms
    noise /= np.std(noise)

    noise = _copy_to_dim(noise, n_channels)

    return noise


def generate_uncorr_noise(duration, fs, n_channels=2, corr=0, seed=None):
    r"""Generate partly uncorrelated noise

    This function generates partly uncorrelated noise using the N+1
    generator method.

    To generate N partly uncorrelated noises with a desired
    correlation coefficent of $\rho$, the algoritm first generates N+1
    noise tokens which are then orthogonalized using the Gram-Schmidt
    process (as implementd in numpy.linalg.qr). The N+1 th noise token
    is then mixed with the remaining noise tokens using the equation

    .. math:: X_{\rho,n} = X_{N+1}  \sqrt{\rho} + X_n  \beta \sqrt{1 - \rho}

    where :math:`X_{\rho,n}` is the nth output and noise,
    :math:`X_{n}` the nth indipendent noise and :math:`X_{N=1}` is the
    common noise.

    for two noise tokens, this is identical to the assymetric
    three-generator method described in [1]_

    Parameters
    ----------
    duration : scalar
        Noise duration in seconds
    fs : int
        Sampling frequency
    n_channels : int
        number of indipendant noise channels
    corr : int, optional
        Desired correlation of the noise tokens, (default=0)
    seed : int or 1-d array_like, optional
        Seed for `RandomState`.
        Must be convertible to 32 bit unsigned integers.

    Returns
    -------
    ndarray
        noise vector of the shape (NxM) where N is the number of samples
        and M >the number of channels

    References
    ----------

    .. [1] Hartmann, W. M., & Cho, Y. J. (2011). Generating partially
      correlated noise—a comparison of methods. The Journal of the
      Acoustical Society of America, 130(1),
      292–301. http://dx.doi.org/10.1121/1.3596475

    """
    np.random.seed(seed)

    sign = np.sign(corr)
    corr = np.abs(corr)
    # if more then one dimension in n_channels
    if np.ndim(n_channels) > 0:
        shape = n_channels
        n_channels = np.product(n_channels)
    else:
        shape = n_channels

    # correlated noise in multiple channels is generated by using the
    # N+1 generator method

    len_signal = nsamples(duration, fs)
    noise = np.random.randn(len_signal, n_channels+1)
    noise -= noise.mean(axis=0)
    # normalize variance
    noise /= noise.std(axis=0)

    # Orthogonalize the noise tokens
    Q, R = np.linalg.qr(noise, 'reduced')

    # normalizing the individual noise energies somewhat reduces
    # the trial-by-trial variance of correlation values
    Q /= Q.std(axis=0)

    # The common noise component is mixed with each of the indebendent
    # noise components to reach the desired correlation
    common_noise = Q[:, -1]
    independent_noise = Q[:, :-1]
    #
    alpha = np.sqrt(corr)
    beta = np.sqrt(1 - alpha**2)
    res_noise = (common_noise.T * alpha + independent_noise.T * beta).T

    # Again make sure that the output variance is 1
    res_noise /= res_noise.std(axis=0)

    # bring into correct shape
    if np.ndim(shape) > 0:
        full_shape = [len(res_noise), *shape]
        res_noise = res_noise.reshape(full_shape)
    # if really only 1 dimensional, return vector
    elif res_noise.shape[1] == 1:
        res_noise = np.squeeze(res_noise)

    return res_noise


def generate_tone(duration, frequency, fs=None, start_phase=0):
    r"""create a cosine

    This function will generate a pure tone following the equation:

    .. math:: x = x + cos(2\pi f t + \phi_0)

    where :math:`x` is the waveform, :math:`f` is the frequency,
    :math`t` is the time and :math:`\phi_0` the starting phase.
    The first evulated timepoint is 0.

    Parameters
    ----------
    duration : scalar
        The tone duration in seconds.
    frequency : scalar
        The tone frequency in Hz.
    fs : scalar
        The sampling rate for the tone.
    start_phase : scalar, optional
        The starting phase of the sine tone.

    Returns
    -------
    ndarray : The sine tone

    See Also
    --------
    audiotools.Signal.add_tone

    """

    duration, fs, ndim = _duration_is_signal(duration, fs, None)

    time = get_time(duration, fs)
    tone = np.cos(2 * pi * frequency * time + start_phase)

    if ndim is not None:
        tone = _copy_to_dim(tone, ndim)

    return tone


def get_time(duration, fs=None):
    r"""Time axis of a given signal.

    This function generates a time axis for a given signal at a given
    sample rate.

    Parameters
    -----------
    duration : ndarray or int
        The duration of the stimulus

    fs : scalar
        The sampling rate in Hz

    Returns
    --------
    ndarray : The time axis in seconds

    """

    duration, fs, _ = _duration_is_signal(duration, fs=fs)

    dt = 1. / fs
    nsamp = nsamples(duration, fs)

    time = np.arange(nsamp) * dt

    return time


def cosine_fade_window(duration, rise_time, fs=None):
    r"""Raised cosine fade-in and fade-out window.

    This function generates a raised cosine / hann fade-in and fade
    out. The Window ramps are calculated as

    .. math:: \frac{1}{2} \left(1 + \cos{\left(\frac{\pi t}{t_r}\right)} \right)

    where :math:`t_r` is the rise_time

    Parameters
    -----------
    duration: ndarray or Signal
        The duration of the stimulus or Signal class
    rise_time : scalar
        Duration of the cosine fade in and fade out in seconds. The
        number of samples is determined via rounding to the nearest
        integer value.
    fs : scalar, optional
        The sampling rate in Hz, is ignored when Signal is passed

    Returns
    -------
    ndarray : The fading window

    """

    duration, fs, ndim = _duration_is_signal(duration, fs, None)

    n_samples = nsamples(duration, fs)
    r = nsamples(rise_time, fs)
    window = np.ones(n_samples)
    flank = 0.5 * (1 + np.cos(pi / r * (np.arange(r) - r)))
    window[:r] = flank
    window[-r:] = flank[::-1]

    # If the signal has multiple channels, extend the window to match
    # the shape
    window = _copy_to_dim(window, ndim)

    return window


def gaussian_fade_window(duration, rise_time, fs=None, cutoff=-60):
    r"""Gausiapn fade-in and fade-out window.

    This function generates a window function with a gausian fade in
    and fade out. The gausian slope is cut at the level defined by the
    cutoff parameter

    The window is given by:

    .. math:: w(t) = e^{\frac{-(t-t_r)^2}{2 * \sigma^2}}

    where :math:`t` is the time, :math:`t_r` is the the rise time and
    :math:`\sigma` is calculated as

    .. math:: \sigma = \sqrt{\frac{r_t^2}{2 \log{(10^{ p / 20})}}}

    where :math:`p` is the cutoff in dB

    Parameters
    -----------
    signal: ndarray, or Signal
        The length of the array will be used to determin the window length.
    rise_time : scalar
        Duration of the gaussian fade in and fade out in seconds. The
        value is measured from the cutof level until reaching a value
        of 1. The number of samples is determined via rounding to the
        nearest integer value.
    fs : scalar
        The sampling rate in Hz
    cutoff : scalar, optional
        The level at which the gausian slope is cut (default = -60dB),
        is ignored when signal is passed

    Returns
    -------
    ndarray : The fading window

    """

    duration, fs, ndim = _duration_is_signal(duration, fs, None)
    n_samples = nsamples(duration, fs)
    window = np.ones(n_samples)

    cutoff_val = 10**(cutoff / 20)  # value at which to cut gaussian
    r = int(np.round(rise_time * fs)) + 1  # number of values in window
    win_time = np.linspace(0, rise_time, r)
    sigma = np.sqrt((-(rise_time)**2 / np.log(cutoff_val)) / 2)
    flank = np.exp(-(win_time - rise_time)**2 / (2 * sigma**2))

    # Set the beginning and and to the window to equal the flank
    window[:r-1] = flank[:-1]
    window[-r:] = flank[::-1]

    window = _copy_to_dim(window, ndim)
    return window


def zeropad(signal, number):
    r"""Add a number of zeros to both sides of a signal

    This function adds a given number of zeros to the start or
    end of a signal.

    If number is a scalar, an equal number of zeros will be appended
    at the front and end of the array. If a vector of two values is
    given, the first defines the number at the beginning, the second
    the number or duration of zeros at the end.

    Parameters
    -----------
    signal: ndarray
        The input Signal
    number : scalar or vecor of len(2), optional
        Number of zeros.

    Returns
    --------
    ndarray : The bufferd signal

    See Also
    --------
    audiotools.Signal.zeropad

    """

    duration, fs, ndim = _duration_is_signal(signal, 1, None)

    if not np.isscalar(number):
        buf_s = np.zeros(number[0])
        buf_e = np.zeros(number[1])
    else:
        buf_s = buf_e = np.zeros(number)

    buf_s = _copy_to_dim(buf_s, ndim)
    buf_e = _copy_to_dim(buf_e, ndim)

    signal_out = np.concatenate([buf_s, signal, buf_e])

    return signal_out


def shift_signal(signal, nr_samples):
    r"""Shift `signal` by `nr_samples` samples.

    Shift a signal by a given number of samples. The shift happens
     cyclically so that the length of the signal does not change.

    Parameters
    ----------
    signal : array_like
        Input signal
    nr_samples : int
        The number of samples that the signal should be shifted. Must
        be positive if `mode` is 'zeros'.

    Returns
    --------
    res : ndarray
        The shifted signal

    See Also:
    ---------
    delay_signal : A high level delaying / shifting function.
    fftshift_signal : Shift a signal in frequency space.

    """
    assert isinstance(nr_samples, int)

    if nr_samples == 0:
        return signal

    sig = np.roll(signal, nr_samples, axis=0)

    return sig

# def fftshift_signal(signal, delay, fs):
#     r"""Delay the `signal` by time `delay` in the frequncy domain.

#     Delays a signal by introducing a linear phaseshift in the
#     frequency domain. Depending on the `mode` this is done cyclically
#     or by zero zeros buffering the start of the signal.

#     Parameters
#     ----------
#     signal : array_like
#         Input signal
#     delay : scalar
#         The delay in seconds. Must be positive if `mode` is 'zeros'.
#     fs : scalar
#         The sampling rate in Hz.

#     Returns
#     --------
#     res : ndarray
#         The shifted signal

#     See Also:
#     ---------
#     delay_signal : A high level delaying / shifting function.
#     shift_signal : Shift a signal by whole samples.

#     """

#     warn("fftshift is depricated",
#          DeprecationWarning)


#     if delay == 0:
#         return signal

#     n_pad = 0
#     len_sig = len(signal)

#     #Apply FFT
#     ft_signal = np.fft.fft(signal, axis=0)

#     #Calculate the phases need for shifting and apply them to the
#     #spectrum
#     freqs = np.fft.fftfreq(len_sig, 1. / fs)
#     phase = time2phase(delay, freqs)
#     ft_signal *= np.exp(-1j * phase)

#     #Inverse transform the spectrum and leave away the imag. part if
#     #it is really small
#     shifted_signal = np.fft.ifft(ft_signal)
#     shifted_signal = np.real_if_close(shifted_signal, 1000)

#     return shifted_signal

def delay_signal(signal, delay, fs, method='fft', mode='zeros'):

    if delay < 0:
        neg_delay = True
        delay = np.abs(delay)
    else:
        neg_delay = False

    # save the original length of the signal
    len_sig = len(signal)

    # due to the cyclic nature of the shift, pad the signal with
    # enough zeros
    n_pad = int(np.ceil(np.abs(delay * fs)))
    pad = np.zeros(n_pad)
    signal = np.concatenate([pad, signal, pad])

    # Apply FFT
    signal = pad_for_fft(signal)
    ft_signal = np.fft.fft(signal)

    # Calculate the phases need for shifting and apply them to the
    # spectrum
    freqs = np.fft.fftfreq(len(ft_signal), 1. / fs)
    ft_signal *= np.exp(-1j * 2 * pi * delay * freqs)

    # Inverse transform the spectrum and leave away the imag. part if
    # it is really small
    shifted_signal = np.fft.ifft(ft_signal)
    shifted_signal = np.real_if_close(shifted_signal, 1000)

    both = np.column_stack([signal, shifted_signal])

    # cut away the buffering
    both = both[n_pad:len_sig + 2 * n_pad, :]

    # If negative delay then just invert the two signals
    if neg_delay:
        both = both[:, ::-1]

    return both


def calc_dbspl(signal):
    r"""Calculate the dB (SPL) value for a given signal.

    .. math:: L = 20  \log_{10}\left(\frac{\sigma}{p_o}\right)

    where :math:`L` is the SPL, :math:`p_0=20\mu Pa` and
    :math:`\sigma` is the RMS of the signal.

    Parameters
    ----------
    signal : ndarray
        The input signal

    Returns
    -------
    float :
        The dB (SPL) value

    See Also
    --------
    audiotools.set_dbspl
    audiotools.Signal.calc_dbspl
    audiotools.Signal.set_dbfs
    audiotools.Signal.calc_dbfs

    """
    p0 = 20e-6
    if np.ndim(signal) != 0:
        rms_val = np.sqrt(np.mean(signal**2, axis=0))
    else:
        rms_val = signal
    dbspl_val = 20 * np.log10(rms_val / p0)

    return dbspl_val


def set_dbspl(signal, dbspl_val):
    r"""Adjust signal amplitudes to a given dbspl value.

    Normalizes the signal to a given sound pressure level in dB
    relative 20e-6 Pa.
    for this, the Signal is multiplied with the factor :math:`A`

    .. math:: A = \frac{p_0}{\sigma} 10^{L / 20}

    where :math:`L` is the goal SPL, :math:`p_0=20\mu Pa` and
    :math:`\sigma` is the RMS of the signal.

    Parameters
    ----------
    signal : ndarray
        The input signal
    dbspl_val : float
        The dbspl value to reach

    Returns
    -------
    ndarray :
        The amplitude adjusted signal

    See Also
    --------
    audiotools.calc_dbspl
    audiotools.Signal.calc_dbspl
    audiotools.Signal.set_dbfs
    audiotools.Signal.calc_dbfs

    """

    if np.ndim(signal) != 0:
        rms_val = np.sqrt(np.mean(signal**2, axis=0))
    else:
        rms_val = signal

    p0 = 20e-6 # ref_value

    factor = (p0 * 10**(float(dbspl_val) / 20)) / rms_val

    return signal * factor


def set_dbfs(signal, dbfs_val):
    r"""Full scale normalization of the signal.

    Normalizes the signal to dB Fullscale
    for this, the Signal is multiplied with the factor :math:`A`

    .. math:: A = \frac{1}{\sqrt{2}\sigma} 10^\frac{L}{20}

    where :math:`L` is the goal Level, and :math:`\sigma` is the
    RMS of the signal.

    Parameters
    ----------
    signal : ndarray
        The input signal
    dbfs_val : float
        The db full scale value to reach

    Returns
    -------
    ndarray :
        The amplitude adjusted signal

    See Also
    --------
    audiotools.set_dbspl
    audiotools.set_dbfs
    audiotools.calc_dbfs
    audiotools.Signal.set_dbspl
    audiotools.Signal.calc_dbspl
    audiotools.Signal.calc_dbfs

    """

    rms0 = 1 / np.sqrt(2)

    if np.ndim(signal) != 0:
        rms_val = np.sqrt(np.mean(signal**2, axis=0))
    else:
        rms_val = signal

    factor = (rms0 * 10**(float(dbfs_val) / 20)) / rms_val

    return signal * factor


def calc_dbfs(signal):
    r"""Calculate the dBFS RMS value of a given signal.

    .. math:: L = 20 \log_{10}\left(\sqrt{2}\sigma\right)

    where :math:`\sigma` is the signals RMS.

    Parameters
    ----------
    signal : ndarray
        The input signal

    Returns
    -------
    float :
        The dBFS RMS value

    """

    rms0 = 1 / np.sqrt(2)

    if np.ndim(signal) != 0:
        rms_val = np.sqrt(np.mean(signal**2, axis=0))
    else:
        rms_val = signal

    dbfs = 20 * np.log10(rms_val / rms0)

    return dbfs


def get_bark_limits():
    r"""Limits of the Bark scale

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

    """
    bark_table = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                  1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                  4400, 5300, 6400, 7700, 9500, 12000, 15500]
    return bark_table


def freqspace(min_frequency, max_frequency, n, scale='bark'):
    r"""Calculate a given number of frequencies that eare equally spaced on
        the bark or erb scale.

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

    """

    if scale == 'bark':
        min_bark, max_bark = freq_to_bark(np.array([min_frequency,
                                                    max_frequency]))
        barks = np.linspace(min_bark, max_bark, n)
        freqs = bark_to_freq(barks)
    elif scale == 'erb':
        min_erb, max_erb = freq_to_erb(np.array([min_frequency,
                                                 max_frequency]))
        erbs = np.linspace(min_erb, max_erb, n)
        freqs = erb_to_freq(erbs)
    else:
        raise NotImplementedError('only ERB and Bark implemented')

    return freqs


def freqarange(min_frequency, max_frequency, step=1, scale='bark'):
    r"""Calculate a of frequencies with a predifined spacing on the bark or erb
        scale.

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

    """
    if scale == 'bark':
        min_bark, max_bark = freq_to_bark(np.array([min_frequency,
                                                    max_frequency]))
        bark = np.arange(min_bark, max_bark, step)
        freqs = bark_to_freq(bark)
    elif scale == 'erb':
        min_erb, max_erb = freq_to_erb(np.array([min_frequency,
                                                 max_frequency]))
        erbs = np.arange(min_erb, max_erb, step)
        freqs = erb_to_freq(erbs)
    else:
        raise NotImplementedError('only ERB and Bark implemented')

    return freqs


def bark_to_freq(bark):
    r"""Bark to frequency conversion

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
    ..[1] Traunmueller, H. (1990). Analytical expressions for the
           tonotopic sensory scale. The Journal of the Acoustical
           Society of America, 88(1),
           97-100. http://dx.doi.org/10.1121/1.399849

    """

    # reverse apply corrections
    bark[bark < 2.0] = (bark[bark < 2.0] - 0.3) / 0.85
    bark[bark > 20.1] = (bark[bark > 20.1] + 4.422) / 1.22
    f = 1960 * (bark + 0.53) / (26.28 - bark)
    return f


def freq_to_bark(frequency, use_table=False):
    r"""Frequency to Bark conversion

    Converts a given sound frequency in Hz into the Bark scale using
    The equation by [2]_ or the original table by [1]_.

    Parameters
    ----------
    frequency: scalar or ndarray
        The frequency in Hz. Value has to be between 20 and 15500 Hz
    use_table: bool, optional
        If True, the original table by [1]_ instead of the equation by
        [2]_ is used. This also results in the CB beeing returned as
        integers.  (default = False)

    Returns
    -------
    scalar or ndarray : The Critical Bandwith in bark

    References
    ----------
    ..[1] Zwicker, E. (1961). Subdivision of the audible frequency
           range into critical bands (frequenzgruppen). The Journal of
           the Acoustical Society of America, 33(2),
           248-248. http://dx.doi.org/10.1121/1.19086f30

    ..[2] Traunmueller, H. (1990). Analytical expressions for the
           tonotopic sensory scale. The Journal of the Acoustical
           Society of America, 88(1),
           97-100. http://dx.doi.org/10.1121/1.399849

    """
    assert np.all(frequency >= 20)
    assert np.all(frequency < 15500)

    if use_table:
        # Only use the table with no intermdiate values
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
    r"""Frequency to number of ERBs conversion

    Calculates the number of erbs for a given sound frequency in Hz
    using the equation by [1]_

    Parameters
    ----------
    frequency: scalar or ndarray
        The frequency in Hz.

    Returns
    -------
    scalar or ndarray : The number of erbs corresponding to the
    frequency

    References
    ----------
    ..[2] Glasberg, B. R., & Moore, B. C. (1990). Derivation of
          auditory filter shapes from notched-noise data. Hearing
          Research, 47(1-2), 103-138.

    """

    n_erb = ((1000. / (24.7 * 4.37))
             * np.log(4.37 * frequency / 1000 + 1))
    return n_erb


def erb_to_freq(n_erb):
    r"""number of ERBs to Frequency conversion

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
    ..[2] Glasberg, B. R., & Moore, B. C. (1990). Derivation of
          auditory filter shapes from notched-noise data. Hearing
          Research, 47(1-2), 103-138.

    """
    fkhz = (np.exp(n_erb * (24.7 * 4.37) / 1000) - 1) / 4.37
    return fkhz * 1000


def phon_to_dbspl(frequency, l_phon, interpolate=False, limit=True):
    r"""Sound pressure levels from loudness level (following DIN ISO 226:2006-04)

    Calulates the sound pressure level at a given frequency that is
    necessary to reach a specific loudness level following DIN ISO
    226:2006-04

    The normed values are tabulated for the following frequencies and
    sound pressure levels:

     1. 20phon to 90phon
       * 20 Hz, 25 Hz, 31.5 Hz, 40 Hz, 50 Hz, 63 Hz, 80 Hz, 100 Hz,
         125 Hz, 160 Hz, 200 Hz, 250 Hz, 315 Hz, 400 Hz, 500 Hz, 630
         Hz, 800 Hz, 1000 Hz, 1250 Hz, 1600 Hz, 2000 Hz, 2500 Hz, 3150
         Hz, 4000 Hz

     2. 20phon to 80phon
       * 5000 Hz, 6300 Hz, 8000 Hz, 10000 Hz, 12500 Hz

    Values for other frequencies can be interpolated (cubic spline) by
    setting the parameter `interpolate=True`. The check for correct
    sound pressure levels can be switched off by setting
    `limit=False`. In both cases, the results are not covered by the
    DIN ISO norm

    Parameters
    ----------
    frequency : scalar
        The frequency in Hz. must be one of the tabulated values above
        if interpolate = False
    l_phon : scalar
        loudness level that should be converted
    interpolate : bool, optional
        Defines whether the tabulated values from the norm should be
        interpolated.  If set to True, the tabulated values will be
        interpolated using a cubic spline (default = False)
    limit : bool, optional
        Defines whether the limits of the norm should be checked
        (default = True)

    Returns
    -------
    The soundpressure level in dB SPL : scalar

    """
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

    if interpolate is False:
        assert frequency in frequency_list
        n_param = np.where(frequency_list == frequency)[0][0]

        alpha_f = alpha_f_list[n_param]
        l_u = l_u_list[n_param]
        t_f = t_f_list[n_param]
    else:
        i_type = 'cubic'
        alpha_f = interp1d(frequency_list,
                           alpha_f_list, kind=i_type)(frequency)
        l_u = interp1d(frequency_list,
                       l_u_list, kind=i_type)(frequency)
        t_f = interp1d(frequency_list,
                       t_f_list, kind=i_type)(frequency)

    a_f = (4.47e-3 * (10**(0.025 * l_phon) - 1.15)
           + (0.4 * 10**((t_f + l_u) / 10 - 9))**alpha_f)
    l_pressure = 10 / alpha_f * np.log10(a_f) - l_u + 94

    return l_pressure


def dbspl_to_phon(frequency, l_dbspl, interpolate=False, limit=True):
    r"""loudness levels from sound pressure level (following DIN ISO 226:2006-04)

    Calulates the loudness level at a given frequency from the sound
    pressure level following DIN ISO 226:2006-04

    The normed values are tabulated for the following frequencies and
    sound pressure levels:
     1. 20phon to 90phon
       * 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
       * 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000
     2. 20phon to 80phon
       *5000, 6300, 8000, 10000, 12500

    Values for other frequencies can be interpolated (cubic spline) by
    setting the parameter interpolate to True. The check for correct
    sound pressure levels can be switched off by setting
    limit=False. In poth cases, the results are not covered by the DIN
    ISO norm

    Parameters
    ----------
    frequency : scalar
        The frequency in Hz. must be one of the tabulated values above
        if interpolate = False
    l_dbspl : scalar
        sound pressure level that should be converted
    interpolate : bool, optional
        Defines whether the tabulated values from the norm should be
        interpolated.  If set to True, the tabulated values will be
        interpolated using a cubic spline (default = False)
    limit : bool, optional
        Defines whether the limits of the norm should be checked
        (default = True)

    Returns
    -------
    scalar : The loudnes level level in dB SPL

    """
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

    if interpolate is False:
        assert frequency in frequency_list
        n_param = np.where(frequency_list == frequency)[0][0]

        alpha_f = alpha_f_list[n_param]
        l_u = l_u_list[n_param]
        t_f = t_f_list[n_param]
    else:
        i_type = 'cubic'
        alpha_f = interp1d(frequency_list,
                           alpha_f_list, kind=i_type)(frequency)
        l_u = interp1d(frequency_list,
                       l_u_list, kind=i_type)(frequency)
        t_f = interp1d(frequency_list,
                       t_f_list, kind=i_type)(frequency)

    b_f = ((0.4 * 10**((l_dbspl + l_u) / 10 - 9))**alpha_f
           - (0.4 * 10**((t_f + l_u) / 10 - 9))**alpha_f + 0.005135)
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
    r"""Calculate approximation of auditory filter bandwidth

    This Function calculates aproximations for the auditory filter
    bandwidth using differnt concepts:

     - cbw: Use the critical bandwidth concept following [1]_
     - erb: Use the equivalent rectangular bandwith concept following
       [2]_

    Equation used for critical bandwidth:
    .. math:: B = 25 + 75 (1 + 1.4 \frac{f_c}{1000}^2)^0.69

    Equation used for critical equivalent rectangular bandwith:
    .. math:: B = 24.7 (4.37 \frac{f_c}{1000} + 1)

    Parameters
    -----------
    fc : float or ndarray
      center frequency in Hz

    scale : str
      String indicating the scale that should be used possible values:
      'cbw' or 'erb'. (default='cbw')

        ..[1] Zwicker, E., & Terhardt, E. (1980). Analytical
              expressions for critical-band rate and critical
              bandwidth as a function of frequency. The Journal of the
              Acoustical Society of America, 68(5), 1523-1525.

        ..[2] Glasberg, B. R., & Moore, B. C. (1990). Derivation of
              auditory filter shapes from notched-noise data. Hearing
              Research, 47(1-2), 103-138.

    """

    if 'cbw' in scale:
        bw = 25 + 75 * (1 + 1.4 * (fc / 1000.)**2)**0.69
    elif 'erb' in scale:
        bw = 24.7 * (4.37 * (fc / 1000.) + 1)

    return bw


def extract_binaural_differences(signal, log_ilds=True):
    r"""Extract the binaural differences between two narrowband signals

    This function extimates the binaural evelope difference as well as the
    phase difference by applying the hilbert transform.

    The envelope difference is defined as the hilbert envelope of the
    first signal minus the hilbert envelope of the second signal while
    the phase difference is defined as the hilbert phase of the first
    minus the hilbert phase of the second.

    Due to the use of a hilbert transform, this approach should only
    be used on signals with a relatively narrow bandwidth.

    Parameters
    -----------
    signal1 : ndarray
        The input signal
    log_ilds : bool, optional
        Defines whether the envelope difference is returned in db
        default = True

    Returns
    -------
    ipd : ndarray
        The phase difference
    env_diff : ndarray
        The envelope difference

    """

    if not isinstance(signal, Signal):
        sig = Signal(2, len(signal), 1)
        sig[:] = signal.copy()
    elif signal.n_channels == 1:
        sig = Signal(2, len(signal), 1)
        sig[:] = signal.copy()[:, None]
    else:
        sig = signal.copy()

    asig = sig.to_analytical()
    ia_sig = asig.ch[0] / asig.ch[1]
    ipd = np.angle(ia_sig)
    ild = np.abs(ia_sig)

    if log_ilds:
        ild = 20 * np.log10(ild)

    if not isinstance(signal, Signal):
        ipd = np.asarray(ipd)
        ild = np.asarray(ild)

    return ipd, ild


def schroeder_phase(harmonics, amplitudes, phi0=0.):
    r"""Phases for a schroeder phase harmonic complex

    This function calculates the phases for a schroeder phase harmonic
    comlex following eq. 11 of [1]_:

    .. math:: \phi_n = \phi_l - 2\pi \sum\limits^{n-1}_{l=1}(n - l)p_l

    :math:`n` is the order of the harmonic and p_l is the relative
    power of the spectral component p_l.

    Parameters
    ----------
    harmonics : ndarray
        A vector of harmonics for which the schroeder phases should be
        calculated
    amplitudes : ndarray
        A vector of amplitudes for the given harmonics
    phi0 : scalar
        The starting phase of the first harmonic (default = 0)

    Returns
    -------
    The phase values for the harmonic compontents : ndarray


    References
    ----------
    .. [1] Schroeder, M. (1970). Synthesis of low-peak-factor signals
          and binary sequences with low autocorrelation
          (corresp.). IEEE Transactions on Information Theory, 16(1),
          85-89

    """
    harmonics = np.array(harmonics)
    amplitudes = np.array(amplitudes)
    power = 0.5 * amplitudes**2
    power /= power.sum()

    phi_schroeder = np.zeros(len(harmonics))
    for i_n, n in enumerate(harmonics):
        phi_shift = 2 * pi * np.sum((n - harmonics[:i_n])
                                    * power[:i_n])
        phi_schroeder[i_n] = phi0 - phi_shift

    return phi_schroeder


def crest_factor(signal, axis=0):
    r"""Calculate crest factor

    Calculates the crest factor of the input signal. The crest factor
    is defined as:

    .. math:: C = \frac{|x_{peak}|}{x_{rms}}

    where :math:`x_{peak}` is the maximum of the absolute value and
    :math:`x_{rms}` is the effective value of the signal.

    Parameters
    -----------
    signal : ndarray
        The input signal
    axis : int
        The axis for which to calculate the crest factor (default = 0)

    Returns
    -------
    scalar :
        The crest factor

    See Also
    --------
    audiotools.Signal.calc_crest_factor

    """
    a_effective = np.sqrt(np.mean(signal**2, axis=axis))
    a_max = np.max(np.abs(signal), axis=axis)

    # crest_factor = 20*np.log10(a_max / a_effective)

    return a_max / a_effective


def calc_coherence(signal):
    r"""normalized complex valued coherence

    This function calculates the normalized complex valued degree of
    coherence between two signals :math:`f(t)` and :math:`g(t)`. It is
    defined as:

    .. math:: \gamma(tau) = \frac{<f_a(t)g^*_a(t-\tau)>}{\sqrt{<|f_a(t)|^2><|g_a(t)|^2>}}

    where :math:`f_a(t)` is the analytic signals of :math:`f(t)` and
    and :math:`g^*_a(t)` is the complex conjugate of the analytic
    signal of :math:`g(t)`. :math:`<\dots>` symbolizes the mean over
    time.

    Requires an input signal with the shape (N, 2).  If only a
    one-dimensional signal is provided, the auto-coherence function
    where :math:`f(t) = g(t)` is calculated.

    The real part of the complex valued coherence equals the
    normalized cross-correlation.

    Parameters
    ----------
    signal : Signal or ndarray
        The input signal

    Returns
    -------
    The coherence vector: Signal or ndarray

    """
    if not isinstance(signal, Signal):
        sig = Signal(2, len(signal), 1)
        sig[:] = signal.copy()
    elif signal.n_channels == 1:
        sig = Signal(2, len(signal), 1)
        sig[:] = signal.copy()[:, None]
    else:
        sig = signal.copy()

    # calculate analytical signal and its spectrum
    fsig = sig.to_freqdomain().to_analytical()
    asig = fsig.to_timedomain()

    # calculate coherence by convolving first channel with complex
    # conjugate of the second channel (done by multiplying fft)
    coh = ((fsig.ch[0] * fsig.ch[1].conj()) / sig.n_samples**2).to_timedomain()

    # normalize by energy so that we gain the normalized coherence function
    coh /= np.sqrt(np.product(np.mean(np.abs(asig)**2, axis=0), axis=0))

    # if input was an ndarray convert output back to ndarray
    coh[:] = np.roll(coh, coh.n_samples//2, axis=0)

    if not isinstance(signal, Signal):
        coh = np.asarray(coh)
    else:
        coh.time_offset = -coh.n_samples//2 * 1 / coh.fs

    return coh
