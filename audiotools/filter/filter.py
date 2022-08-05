import numpy as np

from .gammatone_filt import gammatone
from .butterworth_filt import butterworth
from .brickwall_filt import brickwall
from .. import audiotools as audio


def bandpass(signal, fc, bw, filter_type, fs=None, **kwargs):
    """Apply a bandpass filter to the Signal.

    This function provieds a unified interface to all bandpass filters
    implemented in audiotools.

    Parameters
    ----------
    signal : ndarray or Signal
      The input signal.
    fc : float
      The center frequency in Hz.
    bw : float
      The bandwidth in Hz
    filter_type : string
      The filter type, 'gammatone', 'butter', 'brickwall'
    fs : None or int
      The sampling frequency, must be provided if not using the Signal
      class.
    **kwargs :
      Further arguments such as 'order' that are passed to the filter
      functions.

    Returns
    -------
    Signal : The filtered Signal

    """
    duration, fs, n_channels = audio._duration_is_signal(signal, fs)

    low_f = fc - bw / 2
    high_f = fc + bw / 2
    if filter_type == 'butter':
        sig_out = butterworth(signal, low_f, high_f, fs,
                              **kwargs)
    elif filter_type == 'gammatone':
        sig_out = gammatone(signal, fc, bw, fs, **kwargs)
    elif filter_type == 'brickwall':
        sig_out = brickwall(signal, low_f, high_f, fs, **kwargs)
    else:
        raise(ValueError, f'Filtertype {filter_type} not implemented.')
        return None

    return sig_out


def lowpass(signal, f_cut, filter_type, fs=None, **kwargs):
    """Apply a lowpass filter to the Signal.

    This function provieds a unified interface to all lowpass filters
    implemented in audiotools.

    Parameters
    ----------
    signal : ndarray or Signal
      The input signal.
    f_cut : float
      The cutoff frequency in Hz
    filter_type : string
      The filter type, 'butter', 'brickwall'
    fs : None or int
      The sampling frequency, must be provided if not using the Signal
      class.
    **kwargs :
      Further arguments such as 'order' that are passed to the filter
      functions.

    Returns
    -------
    Signal : The filtered Signal

    """
    duration, fs, n_channels = audio._duration_is_signal(signal, fs)

    if filter_type == 'butter':
        sig_out = butterworth(signal, None, f_cut, fs,
                              **kwargs)
    elif filter_type == 'brickwall':
        sig_out = brickwall(signal, None, f_cut, fs,
                            **kwargs)
    else:
        raise(ValueError, f'Filtertype {filter_type} not implemented.')
        return None

    return sig_out


def highpass(signal, f_cut, filter_type, fs=None, **kwargs):
    """Apply a highpass filter to the Signal.

    This function provieds a unified interface to all highpass filters
    implemented in audiotools.

    Parameters
    ----------
    signal : ndarray or Signal
      The input signal.
    f_cut : float
      The cutoff frequency in Hz
    filter_type : string
      The filter type, 'butter', 'brickwall'
    **kwargs :
      Further arguments such as 'order' that are passed to the filter
      functions.

    Returns
    -------
    Signal : The filtered Signal

    """
    duration, fs, n_channels = audio._duration_is_signal(signal, fs)

    if filter_type == 'butter':
        sig_out = butterworth(signal, f_cut, None, fs,
                              **kwargs)
    elif filter_type == 'brickwall':
        sig_out = brickwall(signal, f_cut, None, fs,
                            **kwargs)
    else:
        raise(ValueError, f'Filtertype {filter_type} not implemented.')
        return None

    return sig_out
