import numpy as np

from . import gammatone_filt as gamma
from . import butterworth_filt as butter
from . import brickwall_filt as brick
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
        sig_out = butter.butterworth(signal, low_f, high_f, fs,
                                     **kwargs)
    elif filter_type == 'gammatone':
        sig_out = gamma.gammatone(signal, fc, bw, fs, **kwargs)
    elif filter_type == 'brickwall':
        sig_out = brick.brickwall(signal, low_f, high_f, fs, **kwargs)
    return sig_out


def lowpass(signal, f_cut, filter_type, fs=None, **kwargs):
    duration, fs, n_channels = audio._duration_is_signal(signal, fs)

    sig_out = butter.butterworth(signal, None, f_cut, fs **kwargs)

    return sig_out
