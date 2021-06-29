import numpy as np

from . import gammatone_filt as gamma
from . import butterworth_filt as butter
from . import brickwall_filt as brick
from .. import audiotools as audio


def bandpass(signal, fc, bw, filter_type, fs=None, **kwargs):
    duration, fs, n_channels = audio._duration_is_signal(signal, fs)

    low_f = fc - bw / 2
    high_f = fc + bw / 2
    if filter_type == 'butter':
        sig_out = butter.butterworth(signal, low_f, high_f, fs, **kwargs)
    elif filter_type == 'gammatone':
        sig_out = gamma.gammatone(signal, fc, bw, fs, **kwargs)
    elif filter_type == 'brickwall':
        sig_out = brick.brickwall(signal, low_f, high_f, fs, **kwargs)
    return sig_out

def lowpass(signal, f_cut, filter_type, fs=None, **kwargs):
    duration, fs, n_channels = audio._duration_is_signal(signal, fs)

    sig_out = butter.butterworth(signal, None, f_cut, fs **kwargs)

    return sig_out
        
    

