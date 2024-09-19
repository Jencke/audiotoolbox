import numpy as np

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


def from_freq(frequency, use_table=False):
    r"""Frequency to Bark conversion

    Converts a given frequency in Hz into the Bark scale using The equation by
    [Traunmueller1990]_ or the original table by [Zwicker1961]_.

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
    ..[Zwicker1961] Zwicker, E. (1961). Subdivision of the audible frequency
           range into critical bands (frequenzgruppen). The Journal of
           the Acoustical Society of America, 33(2),
           248-248. http://dx.doi.org/10.1121/1.19086f30

    ..[Traunmueller1990] Traunmueller, H. (1990). Analytical expressions for the
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


def to_freq(bark):
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


def bandwidth(fc):
    """Calculate critical-bandwidth.

    This function returns the critical bandwidth following [Zwicker1980]_.

    Parameters
    -----------
    fc : float or ndarray
      center frequency in Hz

    Returns
    -------
    The critical bandwidth in Hz

    ..[Zwicker1980] Zwicker, E., & Terhardt, E. (1980). Analytical expressions
          for critical-band rate and critical bandwidth as a function of
          frequency. The Journal of the Acoustical Society of America, 68(5),
          1523-1525.
    """
    bw = 25 + 75 * (1 + 1.4 * (fc / 1000) ** 2) ** 0.69
    return bw
