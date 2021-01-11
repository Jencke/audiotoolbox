import numpy as np
import audiotools as audio
from audiotools.filter import brickwall, gammatone
import copy

class BaseSignal(np.ndarray):
    r"""
    Attributes:
    -----------
    waveform : ndarray
      The signals waveform
    fs
    n_channels
    n_samples
    duration
    time
    """

    def __new__(cls, n_channels, duration, fs, dtype=float):

        n_samples = audio.nsamples(duration, fs)

        if not np.ndim(n_channels): #if channels is only an integer
            if n_channels == 1:
                obj = super(BaseSignal, cls).__new__(cls, shape=(n_samples),
                                                     dtype=dtype)
            else:
                obj = super(BaseSignal, cls).__new__(cls, shape=(n_samples, n_channels),
                                                     dtype=dtype)
        else:
            obj = super(BaseSignal, cls).__new__(cls, shape=[n_samples] + list(n_channels),
                                                     dtype=dtype)
        obj._fs = fs
        obj.fill(0)
        return obj

    def __array_finalize__(self, obj):
        # If called explicitely, obj = None
        if obj is None: return

        # If it was called after e.g slicing, copy
        # copy sample rate
        self._fs = getattr(obj, '_fs', None)


    # getter to handle the sample rates
    @property
    def fs(self):
        """Get the signals sampling rate"""

        return self._fs

    # getter to handle the number of channels in the signal
    @property
    def n_channels(self):
        """Get the number of channels in the signal"""
        if self.ndim == 1:
            return 1
        elif self.ndim == 2:
            return self.shape[1]
        else:
            return self.shape[1:]

    @property
    def n_samples(self):
        """Get the number of samples in the signal"""
        return self.shape[0]

    @property
    def duration(self):
        """Get the duration of the signal in seconds"""
        duration = self.n_samples / self.fs

        return duration

    @property
    def ch(self):
        return _chIndexer(self)

    def concatenate(self, signal):

        if not isinstance(self.base, type(None)):
            raise RuntimeError('Can only concatenate to a full signal')
        else:
            old_n = self.n_samples
            new_n = old_n + signal.n_samples
            new_shape = list(self.shape)
            new_shape[0] = new_n
            self.resize(new_shape, refcheck=False)
            self[old_n:] = signal
        return self

    def multiply(self, x):
        self *= x
        return self

    def add(self, x):
        self += x
        return self

    def subtract(self, x):
        self -= x
        return self

    def abs(self):
        return np.abs(self)

    def copy(self):
        return copy.deepcopy(self)

class _chIndexer(object):
    def __init__(self, obj):
        self.idx_obj = obj
    def __getitem__(self, key):
        # If only one index is handed over
        if not isinstance(key, tuple):
            key = (key, )
        idx = (slice(None, None, None), ) + key
        return self.idx_obj[idx]
    def __setitem__(self, key, value):
        # If only one index is handed over
        if not isinstance(key, tuple):
            key = (key, )
        idx = (slice(None, None, None), ) + key
        self.idx_obj[idx] = value
        return self.idx_obj
