import scipy.signal as sig
import numpy as np
from .. import audiotools as audio

def _copy_to_dim(array, dim):
    if np.ndim(dim) == 0: dim = (dim,)

    #tile by the number of dimensions
    tiled_array = np.tile(array, (*dim[::-1], 1)).T
    #squeeze to remove axis of lenght 1
    tiled_array = tiled_array

    return tiled_array


def butterworth(signal, low_f, high_f, fs=None, order=2,
                return_states=False, states=None):
    r"""Apply a butterwoth filter

    Applies the cascated second-order sections representation of a
    butterwoth IIR filter.

    To construct a lowpass filter, set `low_f` to `None`. For a
    highpass, set `high_f` to `None`.

    Parameters
    ----------
    low_f : scalar or None
        lower cutoff in Hz
    high_f : scalar or None
        high cutoff in Hz
    fs : scalar
       sampling frequency in Hz
    order : integer, optional
       filter order (default = 2)
    return_states : bool, optional
       Wheather the filter states should be returned. (default=False)
    states : True, None or array_like, optional
        Inital conditions for the filter. if True, the conditions for
        a step response are constructed. if set to None, the inital rest is
        assumed (all 0). Otherwise, expects the inital filter delay
        values.

    Returns
    --------
    ndarray : The filtered signal

    """
    _, fs, _ = audio._duration_is_signal(signal, fs, None)

    sos = design_butterworth(low_f, high_f, fs, order)
    filtered_signal, states = apply_sos(signal, sos, states=states)

    if not return_states:
        return filtered_signal
    else:
        return filtered_signal, states


def design_butterworth(low_f, high_f, fs, order=2):
    r"""Return the coeffiecent of a butterwoth filter.

    Returns the cascated second-order sections representation of a
    butterwoth IIR filter. coefficents are calculated using
    scipy.signal.butter

    To construct a lowpass filter, set low_f to None. For a highpass,
    set high_f to None.

    Parameters:
    ----------
    low_f : scalar or None
        lower cutoff in Hz
    high_f : scalar or None
        high cutoff in Hz
    fs : scalar
       sampling frequency in Hz
    order : integer, optional
       filter order (default = 2)

    Returns:
    --------
    Second-order sections representations of the IIR filter : ndarray

    """
    # Determine filtertype
    if low_f is None and high_f is not None:
        sos = sig.butter(order, high_f, btype='lowpass', output='sos', fs=fs)
    elif low_f is not None and high_f is None:
        sos = sig.butter(order, low_f, btype='highpass', output='sos', fs=fs)
    elif low_f is not None and high_f is not None:
        sos = sig.butter(order, [low_f, high_f], btype='bandpass',
                         output='sos', fs=fs)
    else:
        raise Exception('low_f and/or high_f must be provided')
    return sos


def apply_sos(signal, sos, states=None, axis=0):
    r"""Filter the data along one dimension using second order sections.

    Filter the input data using a digital IIR filter defined by sos.

    Parameters:
    -----------
    signal : array like
        The input signal
    sos : array like
        Array of second-order filter coefficients, must have shape
        (n_sections, 6). Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    states : True, None or array_like, optional
        Inital conditions for the filter. if True, the conditions for
        a step response are constructed. if set to None, the inital rest is
        assumed (all 0). Otherwise, expects the inital filter delay
        values.

    Returns:
    --------
    sig_out : ndarray
        The output of the digital filter
    states : ndarray
        the final filter delay values

    """
    _, _, n_channel = audio._duration_is_signal(signal, None, None)

    # initialize states
    if states is True:
        states = sig.sosfilt_zi(sos)
        dim = signal.shape[1:][::-1]
        states = np.tile(states.T, (*dim, 1, 1)).T
    elif states is None:
        order = sos.shape[0]
        if np.ndim(n_channel) == 0:
            if n_channel == 1:  # only one channel
                shape = [order, 2]
            else:               # more then one channels
                shape = [order, 2, n_channel]
        else:                   # Multiple dimensions
            shape = [order, 2, *n_channel]
        states = np.zeros(shape)

    sig_out, states = sig.sosfilt(sos, signal, zi=states, axis=axis)

    return sig_out, states
