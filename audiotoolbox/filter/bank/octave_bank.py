import numpy as np

from .filterbank import ButterworthBank
from ... import audiotools as audio


def get_edge_frequencies(
    fc: float, oct_fraction: int = 3, base_system: int = 2
) -> tuple:
    """Calculate band edge frequencies following ANSI S1.11-2004.

    Calculates the upper and lower band edge frequencies for a given center
    frequency following Eq. 5 and Eq.6 of ANSI S1.11-2004.

    Parameters
    ----------
    fc : float
      The band center frequency in Hz.
    oct_fraction : int (optional)
      The octave fraction to use e.g. 3 for 1/3 octave filters. (default = 3)
    base_system : int (optional)
      The base system to be used. (default = 2)

    Returns
    --------
      flow, fhigh : The lower and uper band edge frequency in Hz.
    """
    b = oct_fraction
    if base_system == 10:
        gbase = 10 ** (3 / 10)
    elif base_system == 2:
        gbase = 2
    else:
        raise (ValueError("base_system must be 2 or 10"))

    f_l = gbase ** (-1 / (2 * b)) * fc
    f_h = gbase ** (1 / (2 * b)) * fc
    return f_l, f_h


def octave_bank(
    fs: int,
    flow: float = 24.8,
    fhigh: float = 20158.0,
    oct_fraction: int = 3,
    round_to_band: bool = True,
    **kwargs
) -> ButterworthBank:
    """Fractional Octave spaced butterworth filterbank.

    Creates a fractional octave filterbank based on butterworth filters. By
    default, This function will create a 1/3 octave bank. If round_to_band is
    set to True (default), center frequencies will be adjusted to fit ANSI
    S1.11-2004.

    Parameters
    ----------
    fs : int
      sampling frequency
    flow : float (optional)
      Lowest center frequency in Hz. (default = 24.8 Hz)
    fhigh : float (optional)
      Highest center frequency in Hz. (default = 20158 Hz)
    oct_frqction : int (optional)
      The fraction of an octave used to space the filter. e.g. 3 for 1/3 octave
      spacing. (default = 3)
    round_to_band : bool (optional)
      Indicates if the band numbers should be rounde to the next full
      integer. If set to True, this will result in center frequencies following
      ANSI S1.11-2004
    **kwargs
      Further paramters such as filter order to pass to the
      filter.ButterworthBank function. Values can either be an ndarray that
      matches the length of `fc` or a single value in which case this value is
      used for all filters.

    Returns
    -------
      ButterworthBank : The filterbank

    """
    # Calculate the corresponding filter banks
    band_low = audio.freq_to_octband(flow, oct_fraction)
    band_high = audio.freq_to_octband(fhigh, oct_fraction)
    if round_to_band:
        band_low = round(band_low)
        band_high = round(band_high)
    # Equaly space filters between the start end end band and convert to center
    # frequencies
    bands = np.arange(band_low, band_high + 1, 1)
    fc = audio.octband_to_freq(bands, oct_fraction)
    # Calculate lower and upper cut-off frequencies as well as bandwidth
    f_l, f_h = get_edge_frequencies(fc, oct_fraction)
    bw = f_h - f_l
    fbank = audio.filter.bank.create_filterbank(
        fc=fc, bw=bw, filter_type="butter", fs=fs, **kwargs
    )
    return fbank
