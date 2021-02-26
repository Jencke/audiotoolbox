import scipy.signal as sig
import numpy as np

def _copy_to_dim(array, dim):
    if np.ndim(dim) == 0: dim = (dim,)

    #tile by the number of dimensions
    tiled_array = np.tile(array, (*dim[::-1], 1)).T
    #squeeze to remove axis of lenght 1
    tiled_array = tiled_array

    return tiled_array


def butterworth(signal, low_f, high_f, fs, order=2):

    sos = design_butterworth(low_f, high_f, fs, order)
    filtered_signal = apply_sos(signal, sos)

    return filtered_signal


def design_butterworth(low_f, high_f, fs, order=2):
    r"""Return the coeffiecent of a butterwoth filter

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

    #Determine filtertype
    if low_f is None and high_f is not None:
        sos = sig.butter(order, high_f, btype='lowpass', output='sos', fs=fs)
    elif low_f is not None and high_f is None:
        sos = sig.butter(order, low_f, btype='highpass', output='sos', fs=fs)
    elif low_f is not None and high_f is not None:
        sos = sig.butter(order, [low_f, high_f], btype='bandpass', output='sos', fs=fs)
    else:
        raise Exception('low_f and/or high_f must be provided')
    return sos

def apply_sos(signal, sos, states=True, axis=0):
    r"""Filter the data along one dimension using second order sections

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
        a step response are constructed. if None, the inital rest is
        assumed (all 0). Otherwise, expects the inital filter delay
        values.

    Returns:
    --------
    sig_out : ndarray
        The output of the digital filter
    states : ndarray
        the final filter delay values

    """

    #initialize states
    if states is True:
        states = sig.sosfilt_zi(sos)
        dim = signal.shape[1:][::-1]
        states = np.tile(states.T, (*dim, 1, 1)).T

    sig_out, states =sig.sosfilt(sos, signal, zi=states, axis=axis)

    return sig_out, states
