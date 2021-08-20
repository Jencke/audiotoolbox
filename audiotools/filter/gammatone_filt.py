import numpy as np
from .. import audiotools as audio
from numpy import pi
from scipy.signal import lfilter


def design_gammatone(fc, bw, fs, order=4, attenuation_db='erb'):
    """Return the coefficient of a gammatone filter.

    Calculates the filter coefficents for a gammatone filter following
    Eq. 11 and 12 of [hohmann2002b]_.


    Parameters
    ----------
    fc : scalar
      The center frequency of the filter in Hz
    bw : scalar
      The bandwidth of the filter in Hz
    fs : int
      The sample frequency
    order : int
      The filter order (default = 4)
    attenuation_db: scalar or 'erb'
      The attenuation at half bandwidth in dB, when set to 'erb', bw
      is interpreted as the equivalent rectangular bandwidth
      of the filter. (default = 'erb')

    Returns
    -------
    b : ndarray
      The numerator coefficient vector in a 1-D sequence.
    a : ndarray
      The denominator coefficient vector in a 1-D sequence.

    References
    ----------
    ..[hohmann2002b] Hohmann, V., Frequency analysis and synthesis
          using a Gammatone filterbank, Acta Acustica, Vol 88 (2002),
          43 -3442

    """
    # in case the bandwith is stated in equivalent rectangular
    # bandwidth:
    if attenuation_db == 'erb':
        # Using Eq. 14 and 15 [Hohmann2002]
        c = 2 * np.sqrt(2**(1 / order) - 1)
        alpha = ((np.pi * np.math.factorial(2 * order - 2) * 2**(2 - 2 * order))
                 / np.math.factorial(order - 1)**2)
        bw = c / alpha * bw
        attenuation_db = -3

    phi = pi * bw / fs          # Eq. 12 [Hohmann2002]
    beta = 2 * pi * fc / fs     # Eq. 10 [Hohmann2002]

    alpha = 10**(0.1 * attenuation_db / order)       # Eq. 12 [Hohmann2002]
    p = (-2 + 2 * alpha * np.cos(phi)) / (1 - alpha) # Eq. 12 [Hohmann2002]

    l = -p / 2 - np.sqrt(p**2 / 4 - 1) # Eq. 12 [Hohmann2002]

    coef = l * np.exp(1j * beta)   # Eq. 1 [Hohmann2002]
    factor = 2 * (1 - np.abs(coef))**order

    b = np.array(factor),
    a = np.array([1., -coef])

    return b, a


def gammatonefos_apply(signal, b, a, order, states=None):
    """Process an input signal by applying the filter `order` times.

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
    _, _, n_channel = audio._duration_is_signal(signal, None, None)

    # state shape
    if not states:
        if np.ndim(n_channel) == 0:
            if n_channel == 1:  # only one channel
                shape = [order, 1]
            else:               # more then one channels
                shape = [order, 1, n_channel]
        else:                   # Multiple dimensions
            shape = [order, 1, *n_channel]

    states = np.zeros(shape, dtype=np.complex128)

    # copy results into a new complex Signal or array
    if isinstance(signal, audio.Signal):
        signal_out = audio.Signal(n_channel, signal.duration,
                                  signal.fs, dtype=complex)
    else:
        signal_out = np.zeros_like(signal, dtype=complex)
    signal_out[:] = signal

    for i in range(order):
        # state = states[i, :]
        out, state = lfilter(b, a, signal_out, zi=states[i], axis=0)
        signal_out[:] = out
        states[i] = state[0]
        b = np.ones_like(b)

    return signal_out, states


def gammatone(signal, fc, bw, fs, order=4, attenuation_db='erb',
              return_complex=True):
    """Apply a gammatone filter to the signal.

    Applys a gammatone filter following [Hohmann2002]_ to the input
    signal and returns the filtered signal.

    Parameters
    ----------
    signal : ndarray
        The input signal
    fs : int
      The sample frequency in Hz
    fc : scalar
      The center frequency of the filter in Hz
    bw : scalar
      The bandwidth of the filter in Hz
    order : int
      The filter order (default = 4)
    attenuation_db: scalar or 'erb'
      The attenuation at half bandwidth in dB (default = -3).
      If set to 'erb', the bandwidth is interpreted as the rectangular
      equivalent bw
    return_complex : bool
      Whether the complex filter output or only it's real
      part is returned (default = True)

    Returns
    -------
      The filtered signal.

    References
    ----------
    .. [Hohmann2002] Hohmann, V., Frequency analysis and synthesis
          using a Gammatone filterbank, Acta Acustica, Vol 88 (2002),
          43 -3442

    """
    b, a = design_gammatone(fc, bw, fs, order, attenuation_db)

    out_signal = np.zeros_like(signal, complex)

    out_signal, state = gammatonefos_apply(signal, b, a, order)

    if not return_complex:
        out_signal = out_signal.real

    return out_signal
