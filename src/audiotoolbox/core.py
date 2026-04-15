"""Function based interface to audiotoolbox."""

from typing import Literal, Optional, Union
import numpy as np
from numpy import pi
from scipy.interpolate import interp1d
from scipy.signal import hilbert, get_window

from .signal import Signal, as_signal
from . import filter
from . import din_iso_226

COLOR_R = "#d65c5c"
COLOR_L = "#5c5cd6"


def _copy_to_dim(array, dim):
    if np.ndim(dim) == 0:
        dim = (dim,)
    # tile by the number of dimensions
    tiled_array = np.tile(array, (*dim[::-1], 1)).T
    # make sure that dimensions are only squeezed if the last dimension of the
    # goal dimension does not equal 1
    if not (len(dim) > 1 & dim[-1] == 1):
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


def from_file(filename: str, start: int = 0, stop: Optional[int] = None) -> Signal:
    """
    Read signal from an audio file.

    This function reads a signal from an audio file and returns it as a Signal object.
    The signal can be read from a specific start point and up to a specific stop point.
    The function supports all audio file formats supported by libsndfile, such as WAV,
    FLAC, AIFF, and more.

    Parameters
    ----------
    filename : str
        The path to the audio file to read.
    start : int, optional
        The starting sample index from which to read the signal. Default is 0.
    stop : int, optional
        The stopping sample index up to which to read the signal. If None, the signal
        is read until the end of the file. Default is None.

    Returns
    -------
    Signal
        The Signal object containing the audio data read from the file.

    Raises
    ------
    ValueError
        If the audio file cannot be read or if the file format is not supported.

    Examples
    --------
    Read a signal from a file starting at the beginning:

    >>> sig = from_file("example.wav")

    Read a signal from a file starting at sample index 1000 and stopping at sample index 5000:

    >>> sig = from_file("example.wav", start=1000, stop=5000)

    See Also
    --------
    audiotoolbox.Signal.from_file : Method to load a signal into an existing Signal object.
    """
    from . import Signal
    from .io import readfile

    wv, fs = readfile(filename, start=start, stop=stop)

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
    out_signal[: len(signal)] = signal

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


def generate_low_noise_noise(
    duration: float,
    fc: float,
    bw: float,
    fs: int,
    n_channels: int | tuple = 1,
    n_rep=10,
    seed=None,
):
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

    # Generate initial noise
    noise = Signal(n_channels, duration, fs).add_noise(ntype="white", seed=seed)
    # noise = generate_noise(duration, fs, ntype="white", n_channels=n_ch)

    std = noise.std(axis=0)

    for i in range(n_rep):
        hilb = noise.to_analytical()
        env = abs(hilb)

        # divide through envelope and restrict
        noise /= env
        noise.bandpass(fc - bw / 2, fc + bw / 2, "brickwall")
        noise /= noise.std(axis=0) * std

    return noise


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
    bark_table = [
        20,
        100,
        200,
        300,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
    ]
    return bark_table


def freqspace(min_frequency, max_frequency, n, scale="bark"):
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

    if scale == "bark":
        min_bark, max_bark = freq_to_bark(np.array([min_frequency, max_frequency]))
        barks = np.linspace(min_bark, max_bark, n)
        freqs = bark_to_freq(barks)
    elif scale == "erb":
        min_erb, max_erb = freq_to_erb(np.array([min_frequency, max_frequency]))
        erbs = np.linspace(min_erb, max_erb, n)
        freqs = erb_to_freq(erbs)
    else:
        raise NotImplementedError("only ERB and Bark implemented")

    return freqs


def freqarange(
    min_frequency: float,
    max_frequency: float,
    step: float = 1,
    scale: Literal["bark", "erb", "octave"] = "bark",
) -> np.ndarray:
    r"""Calculate a of frequencies with a predifined spacing on a given frequency
    scale.

    Returns frequencies between min_frequency and max_frequency with
    a given stepsize step on a frequency scale.

    Parameters
    ----------
    min_frequency: float
        minimal frequency in Hz

    max_frequency: float
        maximal frequency in Hz

    step: float
        stepsize on the scale

    scale: str
        scale to use 'bark' or 'erb' or 'octave'. (default='bark')

    Returns
    -------
    ndarray: frequencies spaced following step on respective scale

    """
    if scale == "bark":
        min_bark, max_bark = freq_to_bark(np.array([min_frequency, max_frequency]))
        bark = np.arange(min_bark, max_bark, step)
        freqs = bark_to_freq(bark)
    elif scale == "erb":
        min_erb, max_erb = freq_to_erb(np.array([min_frequency, max_frequency]))
        erbs = np.arange(min_erb, max_erb, step)
        freqs = erb_to_freq(erbs)
    elif scale == "octave":
        n_steps = int(np.log2(max_frequency / min_frequency) / step)
        exponents = step * (np.arange(n_steps) + 1)
        freqs = max_frequency / 2 ** exponents[::-1]
    else:
        raise NotImplementedError("only ERB and Bark implemented")

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


def octband_to_freq(
    band_nr,
    oct_fraction: Literal[1, 2, 3] = 3,
    base_system: int = 10,
    pref_band: bool = True,
):
    """Octave bandnumber to frequency conversion.

    Converts a given octave band number into the center frequency of
    the band using the equation by [1]_.

    Parameters
    ----------
    band_nr: scalar or ndarray
        The octave band number. The band number is normalized so that
        band 30 is 1000 Hz.
    oct_fraction: int
        The fractional octave scale to use. e.g 3 for 1/3 octave bands.
        default = 3
    base_system: 2 or 10
        The base system used for calcuation. default = 10,
    pref_band: bool
        If True, the frequency is rounded to the nearest preferred
        frequency according to ISO 226:2003. (default = True)

    Returns
    -------
    scalar or ndarray: The center frequency of the octave band in Hz.

    References
    ----------
    ..[1] DIN ISO 266-1:1997-08, "Acoustics - Preferred frequencies",
        Beuth Verlag, Berlin, 1997.
    """

    b = oct_fraction

    if base_system == 10:
        gbase = 10 ** (3 / 10)
    elif base_system == 2:
        gbase = 2
    else:
        raise (ValueError("base_system must be 2 or 10"))

    if b % 2:  # if odd
        freq = gbase ** ((band_nr - 30.0) / b) * 1e3
    else:  # if even:
        freq = gbase ** ((2 * band_nr - 59.0) / (2 * b)) * 1e3

    if pref_band:
        freq = din_iso_226.round_array_to_pref_freq(freq)

    return freq


def freq_to_octband(
    frequency, oct_fraction: int = 3, base_system: int = 10, round: bool = True
):
    """Frequency to octave bandnumber conversion.

    Scales are normalized so that band 1000Hz is band 30

    Parameters
    ----------
    frequency: scalar or ndarray
        The frequency in Hz.
    oct_fraction: int
        The fractional octave scale to use. e.g 3 for 1/3 octave bands.
        default = 3
    base_system: 2 or 10
        The base system used for calcuation. default = 2
    round: bool
        If True, the band number is rounded to the nearest integer.
        (default = True)
    """
    b = oct_fraction
    if base_system == 10:
        gbase = 10 ** (3 / 10)
    elif base_system == 2:
        gbase = 2
    else:
        raise (ValueError("base_system must be 2 or 10"))
    if b % 2:
        band_nr = np.log(frequency / 1000) / np.log(gbase) * b + 30
    else:
        band_nr = 0.5 * (np.log(frequency / 1000) / np.log(gbase) * 2 * b + 59)

    if round:
        band_nr = np.round(band_nr, 0)
    return band_nr


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

    n_erb = (1000.0 / (24.7 * 4.37)) * np.log(4.37 * frequency / 1000 + 1)
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
    frequency_list = din_iso_226.frequency_list

    alpha_f_list = din_iso_226.alpha_f_list

    # transfer function normed at 1000Hz
    l_u_list = din_iso_226.l_u_list

    # Hearing threshold t_f
    t_f_list = din_iso_226.t_f_list

    if interpolate is False:
        assert frequency in frequency_list
        n_param = np.where(frequency_list == frequency)[0][0]

        alpha_f = alpha_f_list[n_param]
        l_u = l_u_list[n_param]
        t_f = t_f_list[n_param]
    else:
        i_type = "cubic"
        alpha_f = interp1d(frequency_list, alpha_f_list, kind=i_type)(frequency)
        l_u = interp1d(frequency_list, l_u_list, kind=i_type)(frequency)
        t_f = interp1d(frequency_list, t_f_list, kind=i_type)(frequency)

    a_f = (
        4.47e-3 * (10 ** (0.025 * l_phon) - 1.15)
        + (0.4 * 10 ** ((t_f + l_u) / 10 - 9)) ** alpha_f
    )
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
       * 5000, 6300, 8000, 10000, 12500

    Values for other frequencies can be interpolated (cubic spline) by
    setting the parameter interpolate to True. The check for correct
    sound pressure levels can be switched off by setting
    limit=False. In both cases, the results are not covered by the DIN
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
    frequency_list = din_iso_226.frequency_list
    alpha_f_list = din_iso_226.alpha_f_list

    # transfer function normed at 1000Hz
    l_u_list = din_iso_226.l_u_list

    # Hearing threshold t_f
    t_f_list = din_iso_226.t_f_list

    if interpolate is False:
        assert frequency in frequency_list
        n_param = np.where(frequency_list == frequency)[0][0]

        alpha_f = alpha_f_list[n_param]
        l_u = l_u_list[n_param]
        t_f = t_f_list[n_param]
    else:
        i_type = "cubic"
        alpha_f = interp1d(frequency_list, alpha_f_list, kind=i_type)(frequency)
        l_u = interp1d(frequency_list, l_u_list, kind=i_type)(frequency)
        t_f = interp1d(frequency_list, t_f_list, kind=i_type)(frequency)

    b_f = (
        (0.4 * 10 ** ((l_dbspl + l_u) / 10 - 9)) ** alpha_f
        - (0.4 * 10 ** ((t_f + l_u) / 10 - 9)) ** alpha_f
        + 0.005135
    )
    l_phon = 40 * np.log10(b_f) + 94

    if limit:
        # Definition only valid starting from 20 phon
        assert l_phon >= 20

        if 20 <= frequency <= 4500:
            assert l_phon <= 90
        elif 4500 < frequency <= 12500:
            assert l_phon <= 80

    return l_phon


def calc_bandwidth(fc, scale="cbw"):
    r"""Calculate approximation of auditory filter bandwidth

    This Function calculates aproximations for the auditory filter
    bandwidth using differnt concepts:

    - cbw: Use the critical bandwidth concept following [1]_
    - erb: Use the equivalent rectangular bandwith concept following [2]_

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

    if "cbw" in scale:
        bw = 25 + 75 * (1 + 1.4 * (fc / 1000.0) ** 2) ** 0.69
    elif "erb" in scale:
        bw = 24.7 * (4.37 * (fc / 1000.0) + 1)

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


def schroeder_phase(harmonics, amplitudes, phi0=0.0):
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
        phi_shift = 2 * pi * np.sum((n - harmonics[:i_n]) * power[:i_n])
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
    audiotoolbox.Signal.calc_crest_factor

    """
    a_effective = np.sqrt(np.mean(signal**2, axis=axis))
    a_max = np.max(np.abs(signal), axis=axis)

    # crest_factor = 20*np.log10(a_max / a_effective)

    return a_max / a_effective


def inst_cmplx_corr(signal, window_duration, window="hann"):
    r"""Calculate instantaneous complex correlation

    This function calculates the instantaneous complex correlation of a
    signal using a sliding window approach.

    Parameters
    ----------
    signal : Signal or ndarray
        The input signal
    window_duration : float
        The duration of the sliding window in seconds
    window : str
        The type of window to use (default = 'hann')

    Returns
    -------
    corr : ndarray
        The instantaneous complex correlation

    """

    asig = signal.to_analytical()
    iccp = asig.ch[0] * asig.ch[1].conjugate()
    win_samps = int(window_duration * signal.fs)
    win = as_signal(get_window(window, win_samps), signal.fs)
    filt_iccp = iccp.convolve(win, "same")    
    filt_pow1 = (np.abs(asig.ch[0])**2).convolve(win, "same")
    filt_pow2 = (np.abs(asig.ch[1])**2).convolve(win, "same")
    filt_icpow = np.sqrt(filt_pow1 * filt_pow2)
    coh = filt_iccp / filt_icpow
    return coh


def cmplx_corr(signal, fs=None):
    r"""The complex valued correlation coefficent.

    This function calculates the complex valued correlation coefficent which
    equals the value of the complex_valued_cross_correlation at :math:`\tau=0`

    .. math:: \gamma = \frac{<f_a(t)^*_g_a(t)>}{\sqrt{<|f_a(t)|^2><|g_a(t)|^2>}}

    where :math:`f_a(t)` is the analytic signals of :math:`f(t)` and
    and :math:`g^*_a(t)` is the complex conjugate of the analytic
    signal of :math:`g(t)`. :math:`<\dots>` symbolizes the mean over
    time.

    Parameters
    ----------
    signal : Signal or ndarray
        The input signal. The shape must be (N, 2) where N are the
        samples.

    Returns
    -------
    The coherence vector: Signal or ndarray

    """

    if np.ndim(signal) < 2:
        raise ValueError("Input shape must be (N, 2)")
    if np.shape(signal)[1] != 2:
        raise ValueError("Input shape must be (N, 2)")

    sig = as_signal(signal, fs)
    asig = sig.to_analytical()

    ccm = np.mean(asig.ch[0] * asig.ch[1].conjugate(), axis=0)
    norm1 = np.mean(np.abs(asig.ch[0]) ** 2, axis=0)
    norm2 = np.mean(np.abs(asig.ch[1]) ** 2, axis=0)

    corrcov = ccm / np.sqrt(norm1 * norm2)

    return corrcov


def cmplx_crosscorr(signal):
    r"""normalized complex valued cross-correlation function

    This function calculates the normalized complex valued cross correlation
    function between two signals :math:`f(t)` and :math:`g(t)`. It is defined
    as:

    .. math:: \gamma(tau) = \frac{<f_a(t)^*g_a(t-\tau)>}{\sqrt{<|f_a(t)|^2><|g_a(t)|^2>}}

    where :math:`f_a(t)` is the analytic signals of :math:`f(t)` and
    and :math:`g^*_a(t)` is the complex conjugate of the analytic
    signal of :math:`g(t)`. :math:`<\dots>` symbolizes the mean over
    time.

    Requires an input signal with the shape (N, 2).  If only a one-dimensional
    signal is provided, the auto-cross correlation function where :math:`f(t) =
    g(t)` is calculated.

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
    coh = ((fsig.ch[0] * fsig.ch[1].conj())).to_timedomain()

    # normalize by energy so that we gain the normalized coherence function
    coh /= np.sqrt(np.prod(np.mean(np.abs(asig) ** 2, axis=0), axis=0))

    # if input was an ndarray convert output back to ndarray
    coh[:] = np.roll(coh, coh.n_samples // 2, axis=0)

    if not isinstance(signal, Signal):
        coh = np.asarray(coh)
    else:
        coh.time_offset = -coh.n_samples // 2 * 1 / coh.fs

    return coh


def crossfade(
    sig1: Signal,
    sig2: Signal,
    fade_duration: float,
    fade_type: Literal["linear", "cos"] = "linear",
) -> Signal:
    """Crossfade two Signals

    Apply a crossfade between the end of the first and the beginning of the
    second signal.

    Parameters
    ----------
    sig1 : Signal
        First signal
    sig2 : Signal
        Second signal
    fade_duration : float
        duration of the fade in seconds
    fade_type : Literal["linear", "cos"]
        Type of the crossfade

    Returns
    -------
    Signal
        The resulting signal.
    """
    if sig1.n_channels != sig2.n_channels:
        raise (ValueError("The two signals need to match in number of channels."))
    if sig1.fs != sig2.fs:
        raise (ValueError("The sample rate of the two signals has to match."))
    fs = sig1.fs
    n_channels = sig1.n_channels

    fade = Signal(1, fade_duration, fs)
    if fade_type == "cos":
        fade[:] = np.cos(np.pi / 2 * fade.time / fade_duration)
    elif fade_type == "linear":
        fade[:] = 1 - fade.time / fade.time[-1]
    else:
        raise (ValueError("fade_type not implemented"))

    n_out = sig1.n_samples + sig2.n_samples - fade.n_samples
    out_duration = n_out / fs
    out_sig = Signal((2,) + tuple(np.atleast_1d(n_channels)), out_duration, fs)

    # Reshape signals to ensure they can be broadcast into the temporary out_sig.
    # This is necessary because a 1D signal (N,) cannot be assigned to a
    # 2D slice (N, 1) without an explicit reshape.
    s1 = sig1.reshape(sig1.n_samples, *np.atleast_1d(sig1.n_channels))
    s2 = sig2.reshape(sig2.n_samples, *np.atleast_1d(sig2.n_channels))

    out_sig[: sig1.n_samples, 0] = s1
    out_sig[-sig2.n_samples :, 1] = s2

    fade_s = sig1.n_samples - fade.n_samples
    fade_e = fade_s + fade.n_samples

    # Reshape fade ramps to allow broadcasting across all channel dimensions.
    # A ramp of shape (N,) becomes (N, 1) or (N, 1, 1) etc., to match the
    # shape of the signal slice it's being multiplied with.
    n_channel_dims = len(np.atleast_1d(n_channels))
    fade_shape = (-1,) + (1,) * n_channel_dims
    fade_out_ramp = fade.reshape(fade_shape)
    fade_in_ramp = fade[::-1].reshape(fade_shape)

    out_sig[fade_s:fade_e, 0] *= fade_out_ramp
    out_sig[fade_s:fade_e, 1] *= fade_in_ramp

    out_sig = out_sig.sum(axis=1)
    return out_sig


def _get_dim_overlap(dim1, dim2):
    """
    Calculates the number of matching elements from the end of dim1 and
    the beginning of dim2.

    Parameters
    ----------
    dim1 : tuple or list
        The first sequence of elements.
    dim2 : tuple or list
        The second sequence of elements.

    Returns
    -------
    int
        The maximum number of consecutive elements that match between the
        end of `dim1` and the beginning of `dim2`. If there are no matching
        elements, returns 0.

    Examples
    --------
    >>> get_dim_overlap((1, 2, 3, 4), (3, 4, 5))
    2
    >>> get_dim_overlap((8, 9), (1, 2, 3))
    0
    """
    dim1 = tuple(dim1)
    dim2 = tuple(dim2)
    max_overlap = min(len(dim1), len(dim2))
    for overlap_length in range(max_overlap, 0, -1):
        if dim1[-overlap_length:] == dim2[:overlap_length]:
            return overlap_length
    return 0
