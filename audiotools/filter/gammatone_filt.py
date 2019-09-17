import numpy as np
from numpy import pi
from scipy.signal import lfilter

def design_gammatone(cf, bw, fs, order=4, attenuation_db=-3):
    """Returns the coefficient of a gammatone filter

    Calculates the filter coefficents for a gammatone filter following
    Eq. 11 and 12 of [1]_.


    Parameters
    ----------
    cf : scalar
      The center frequency of the filter in Hz
    bw : scalar
      The bandwidth of the filter in Hz
    fs : int
      The sample frequency
    order : int
      The filter order (default = 4)
    attenuation_db: scalar
      The attenuation at half bandwidth in dB (default = -3)


    Returns
    -------
    b : ndarray
      The numerator coefficient vector in a 1-D sequence.
    a : ndarray
      The denominator coefficient vector in a 1-D sequence.

    References
    ----------
    ..[1] Hohmann, V., Frequency analysis and synthesis using a
          Gammatone filterbank, Acta Acustica, Vol 88 (2002), 43 -3442

    """
    phi = pi * bw / fs          # Eq. 12 [Hohmann2002]
    beta = 2 * pi * cf / fs     # Eq. 10 [Hohmann2002]

    alpha = 10**(0.1 * attenuation_db / order)       # Eq. 12 [Hohmann2002]
    p = (-2 + 2 * alpha * np.cos(phi)) / (1 - alpha) # Eq. 12 [Hohmann2002]

    l = -p / 2 - np.sqrt(p * p / 4 - 1) # Eq. 12 [Hohmann2002]

    coef = l * np.exp(1j * beta)   # Eq. 1 [Hohmann2002]
    factor = 2 * (1 - np.abs(coef))**order

    b = np.array(factor),
    a = np.array([1., -coef])

    return b, a


def gammatonefos_apply(signal, b, a, order, states=None):
    """Process an input signal by applying the filter `order` times

    Filter the signal with a gammatone filter defined by the
    coeffients `b` and `a`. The filter is applied `order` times.

    Parameters
    ----------
    b : array_like
      The numerator coefficient vector in a 1-D sequence.
    a : array_like
      The denominator coefficient vector in a 1-D sequence.
    order : int
      The filter order
    states : ndarray or None
      Filter states of length `order`. (default = none)


    Returns
    -------
    signal : ndarray
      The analytical filtered output signal
    states : ndarray
      The filter states.

    """
    if not states:
        states = np.zeros(order, dtype=np.complex128)

    for i in range(order):
        state = [states[i]]
        signal, state = lfilter(b, a, signal, zi=state)
        states[i] = state[0]
        b = np.ones_like(b)

    return signal, states
