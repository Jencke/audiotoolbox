import numpy as np
from numpy import pi, sqrt, log10
from scipy.signal import zpk2sos, zpk2tf, bilinear_zpk, freqs_zpk
from .. import audiotools as audio
from .butterworth_filt import apply_sos


def calc_analog_poles() -> tuple:
    """Poles of the analog A and C weighting filters following IEC 61672-1."""

    # Equations follow IEC 61672-1 (2002)

    # Cutoff freqs as stated in section 5.4.6
    f_r = 1000  # ref. freq
    f_l = 10**1.5  # low cutoff
    f_h = 10**3.9  # high cutoff
    f_a = 10**2.45  # low cutoff for A weighting

    d = sqrt(1 / 2)
    b = (1 / (1 - d)) * (
        f_r**2 + (f_l**2 * f_h**2) / f_r**2 - d * (f_l**2 + f_h**2)
    )  # Eq 11
    c = f_l**2 * f_h**2  # Eq 12

    f1 = (0.5 * (-b - sqrt(b**2 - 4 * c))) ** 0.5  # Eq. 9
    f4 = (0.5 * (-b + sqrt(b**2 - 4 * c))) ** 0.5  # Eq. 10
    f2 = (3 - sqrt(5)) / 2 * f_a  # Eq. 13
    f3 = (3 + sqrt(5)) / 2 * f_a  # Eq. 14

    return f1, f2, f3, f4


def c_weight(freq):
    """Gain of the C-weighting filter in dB

    Calculates the filter gain of the C-weighting filter following IEC 61672-1.

    Parameters
    ----------
    freq :
      The frequency

    Returns
    -------
      The filter gain in dB.
    """
    f1, f2, f3, f4 = calc_analog_poles()

    def c_weight(f):
        c_weight = 20 * log10(
            (f4**2 * f**2) / ((f**2 + f1**2) * (f**2 + f4**2))
        )
        return c_weight

    c1000 = c_weight(1000)
    c_freq = c_weight(freq) - c1000
    return c_freq


def a_weight(freq):
    """Gain of the A-weighting filter in dB

    Calculates the filter gain of the A-weighting filter following IEC 61672-1.

    Parameters
    ----------
    freq :
      The frequency

    Returns
    -------
    The filter gain in dB.
    """
    f1, f2, f3, f4 = calc_analog_poles()

    def a_weight(f):
        a_weight = 20 * log10(
            (f4**2 * f**4)
            / (
                (f**2 + f1**2)
                * sqrt(f**2 + f2**2)
                * sqrt(f**2 + f3**2)
                * (f**2 + f4**2)
            )
        )
        return a_weight

    a1000 = a_weight(1000)
    a_freq = a_weight(freq) - a1000
    return a_freq


def design_a_filter(fs, ftype="sos"):
    """Digital filter design for A-weighting.

    Designs a digital a-weighting filter following IEC 61672-1

    Parameters
    ----------
    fs : int
        The sampling frequency
    ftype: {'sos', 'zpk', 'ba'}, optional
        The filter coefficents that should be returned. default is 'zpk'

    Returns
    -------
        The filter coefficents. sos in case of sos, (z, p, k) in case of zpk and
        b, a in case of ba.
    """
    f1, f2, f3, f4 = calc_analog_poles()
    z_analog = [0, 0, 0, 0]
    p_analog = np.array([-f1, -f1, -f2, -f3, -f4, -f4]) * 2 * np.pi
    # determine k so that gain is 0 dB at 1Khz
    k_analog = np.abs(freqs_zpk(z_analog, p_analog, 1, worN=[1000 * 2 * pi])[1]) ** -1
    k_analog = k_analog[0]
    #
    z, p, k = bilinear_zpk(z_analog, p_analog, k_analog, fs)

    if ftype == "sos":
        sos = zpk2sos(z, p, k)
        return sos
    elif ftype == "zpk":
        return z, p, k
    elif ftype == "ba":
        b, a = zpk2tf(z, p, k)
        return b, a


def design_c_filter(fs, ftype="sos"):
    """Digital filter design for C-weighting.

    Designs a digital C-weighting filter following IEC 61672-1

    Parameters
    ----------
    fs : int
        The sampling frequency
    ftype: {'sos', 'zpk', 'ba'}, optional
        The filter coefficents that should be returned. default is 'zpk'

    Returns
    -------
        The filter coefficents. sos in case of sos, (z, p, k) in case of zpk and
        b, a in case of ba.
    """
    f1, f2, f3, f4 = calc_analog_poles()
    z_analog = [0, 0]
    p_analog = np.array([-f1, -f1, -f4, -f4]) * 2 * np.pi
    # determine k so that gain is 0 dB at 1Khz
    k_analog = np.abs(freqs_zpk(z_analog, p_analog, 1, worN=[1000 * 2 * pi])[1]) ** -1
    k_analog = k_analog[0]
    #
    z, p, k = bilinear_zpk(z_analog, p_analog, k_analog, fs)

    if ftype == "sos":
        sos = zpk2sos(z, p, k)
        return sos
    elif ftype == "zpk":
        return z, p, k
    elif ftype == "ba":
        b, a = zpk2tf(z, p, k)
        return b, a


def a_weighting(signal, fs=None):
    """Apply A weighting filter to a signal.

    Apply an digital A-weighting filter following IEC 61672-1.
    Take care that the sampling frequency is high enough to prevent steep
    high frequency drop-off. A sampling frequency of about 48 kHz should be
    sufficent to result in a Class 1 filter following IEC 61672-1.

    Parameters:
    -----------
    signal : Signal or np.ndarray
        The input signal
    fs : scalar or None
      The signals sampling rate in Hz.

    Returns
    -------
        The filtered signal.
    """
    _, fs, _ = audio._duration_is_signal(signal, fs, None)

    sos = design_a_filter(fs)
    out, _ = apply_sos(signal, sos, states=True)

    if isinstance(signal, audio.Signal):
        out = audio.as_signal(out, fs)

    return out


def c_weighting(signal, fs=None):
    """Apply C weighting filter to a signal.

    Apply an digital C-weighting filter following IEC 61672-1.
    Take care that the sampling frequency is high enough to prevent steep
    high frequency drop-off. A sampling frequency of about 48 kHz should be
    sufficent to result in a Class 1 filter following IEC 61672-1.

    Parameters:
    -----------
    signal : Signal or np.ndarray
        The input signal
    fs : scalar or None
      The signals sampling rate in Hz.

    Returns
    -------
        The filtered signal.
    """
    _, fs, _ = audio._duration_is_signal(signal, fs, None)

    sos = design_c_filter(fs)
    out, _ = apply_sos(signal, sos, states=True)

    if isinstance(signal, audio.Signal):
        out = audio.as_signal(out, fs)

    return out
