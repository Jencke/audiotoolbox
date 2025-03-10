import numpy as np

def to_freq(band_nr, oct_fraction: int = 3, base_system: int = 2):
    """Octave bandnumber to frequency conversion.

    Converts a (fractional) octave band nr to it's frequency. This can either
    use a base 2 or a base 10 system.

    Parameters
    ----------
    frequency: scalar or ndarray
        The frequency in Hz.
    oct_fraction: int
        The fractional octave scale to use. e.g 3 for 1/3 octave bands.
        default = 3
    base_system: {2, 10}
      The base system used for calcuation. default = 2

    Returns
    -------
    The frequencies
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
    return freq


def from_freq(frequency, oct_fraction: int = 3, base_system: int = 2):
    """Frequency to octave bandnumber conversion.

    Scales are normalized so that band 1000Hz is band 30

    Parameters
    ----------
    frequency: scalar or ndarray
        The frequency in Hz.
    oct_fraction: int
        The fractional octave scale to use. e.g 3 for 1/3 octave bands.
        default = 3
    base_system: {2, 10}
      The base system used for calcuation. default = 2

    Returns
    -------
    The octave band number
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
    return band_nr
