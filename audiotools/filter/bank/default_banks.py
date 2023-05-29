from ... import audiotools as audio
from . filterbank import create_filterbank


def auditory_gamma_bank(fs, flow=16, fhigh=16000, step=1, **kwargs):
    """Equivalent Rectangular Bandwidth spaced gammatone filterbank.

    Creates a gammatone filterbank with center freequencies equally spaced on
    the auditory ERB scale as proposed by [1]_.

    ..[1] Glasberg, B. R., & Moore, B. C. (1990). Derivation of
          auditory filter shapes from notched-noise data. Hearing
          Research, 47(1-2), 103-138.

    Parameters
    ----------
    fs : int
      sampling frequency
    flow : scalar (optional)
      Lowest center frequency in Hz. (default = 16 Hz)
    fhigh : scalar (optional)
      Highest center frequency in Hz. (default = 16 kHz)
    step : scalar (optional)
      Stepsize between filters on the ERB scale. Defauls to 1 filter per ERB
    **kwargs
      Further paramters such as filter order to pass to the
      filter.GammaToneBank function. Values can either be an ndarray that
      matches the length of `fc` or a single value in which case this value is
      used for all filters.

    Returns
    _______
        GammaToneBank : The filterbank.

    """
    fc = audio.freqarange(flow, fhigh, step=step, scale='erb')
    bw = audio.calc_bandwidth(fc, 'erb')
    fbank = create_filterbank(fc, bw, 'gammatone', fs=fs, **kwargs)
    return fbank
