import numpy as np

def to_freq(n_erb):
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


def from_freq(frequency):
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


def bandwidth(fc):
    '''Calculate bandwidth on the ERB scale.

    This function returns the equivalent rectangular bandwidth for a given
    center frequency following [Glasberg1990]_

    Parameters
    -----------
    fc : float or ndarray
      center frequency in Hz

    Returns
    -------
    The ERB in Hz

    ..[Glasberg1990] Glasberg, B. R., & Moore, B. C. (1990). Derivation of
          auditory filter shapes from notched-noise data. Hearing Research,
          47(1-2), 103-138.
    '''
    bw = 24.7 * (4.37 * (fc / 1000) + 1)
    return bw
