import numpy as np
import audiotools as audio


class BaseSignal(np.ndarray):
    r""" Basic Signal class inherited by all Signal representations

    """
    def __new__(cls, n_channels, duration, fs, dtype=float):

        n_samples = audio.nsamples(duration, fs)

        if not np.ndim(n_channels):  # if channels is only an integer
            if n_channels == 1:
                obj = super(BaseSignal, cls).__new__(cls, shape=(n_samples),
                                                     dtype=dtype)
            else:
                obj = super(BaseSignal, cls).__new__(cls, shape=(n_samples,
                                                                 n_channels),
                                                     dtype=dtype)
        else:
            obj = super(BaseSignal, cls).__new__(cls, shape=[n_samples] +
                                                 list(n_channels),
                                                 dtype=dtype)
        obj._fs = fs
        obj.fill(0)
        return obj

    def __array_finalize__(self, obj):
        # If called explicitely, obj = None
        if obj is None:
            return

        # If it was called after e.g slicing, copy
        # copy sample rate
        self._fs = getattr(obj, '_fs', None)

    @property  # getter to handle the sample rates
    def fs(self):
        """Sampling rate of the signal in Hz"""

        return self._fs

    # getter to handle the number of channels in the signal
    @property
    def n_channels(self):
        """Number of channels in the signal"""
        if self.ndim == 1:
            return 1
        elif self.ndim == 2:
            return self.shape[1]
        else:
            return self.shape[1:]

    @property
    def n_samples(self):
        """Number of samples in the signal"""
        return self.shape[0]

    @property
    def duration(self):
        """Duration of the signal in seconds"""
        duration = self.n_samples / self.fs

        return duration

    @property
    def ch(self):
        r"""Direct channel indexer

        Returns an indexer class which enables direct indexing and
        slicing of the channels indipendent of samples.

        Examples
        --------
        >>> sig = audiotools.Signal((2, 3), 1, 48000).add_noise()
        >>> print(np.all(sig.ch[1, 2] is sig[:, 1, 2]))
        True

        """
        return _chIndexer(self)

    def concatenate(self, signal):
        '''Concatenate another signal or array

        This method appends another signal to the end of the current
        signal.

        Parameters
        -----------
        signal : signal or ndarray
            The signal to append

        Returns
        --------
        Returns itself

        '''
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
        """In-place multiplication

        This function allowes for in-place multiplication

        Parameters
        -----------
        x : scalar or ndarray
            The value or array to muliply with the signal

        Returns
        --------
        Returns itself

        Examples
        --------
        >>> sig = audiotools.Signal(1, 1, 48000).add_tone(500).multiply(2)
        >>> print(sig.max())
        2.0

        """
        self *= x
        return self

    def add(self, x):
        """In-place summation

        This function allowes for in-place summation.

        Parameters
        -----------
        x : scalar or ndarray
            The value or array to add to the signal

        Returns
        --------
        Returns itself

        Examples
        --------
        >>> sig = audiotools.Signal(1, 1, 48000).add_tone(500).add(2)
        >>> print(sig.mean())
        2.0

        """

        self += x
        return self

    def abs(self):
        """ Absolute value

        Calculates the absolute value or modulus of all values of the signal

        """
        return np.abs(self)

    def copy_empty(self):
        out = self.copy()
        out[:] = 0
        return out

    def summary(self):
        if self.duration < 1:
            duration = f'{self.duration * 1000:2}ms'
        else:
            duration = f'{self.duration:2}s'

        if self.fs < 1000:
            fs = f"{self.fs}Hz"
        else:
            fs = f"{self.fs / 1000:.1f}kHz"

        samp = f'{self.n_samples} samples'

        chan = f'{self.n_channels} channel'

        repr = (duration
                + ' @ ' + fs
                + " = " + samp
                + ' in ' + chan
                + ' | dtype: ' + str(self.dtype))
        return repr


class _chIndexer(object):
    """Channel Indexer

    Allowes channels to be indexed directly without needing to care about
    samples

    """

    def __init__(self, obj):
        self.idx_obj = obj

    def __getitem__(self, key):

        if not isinstance(key, tuple):
            # If only one index is handed over, convert key to tuple
            key = (key, )

        if np.ndim(self.idx_obj) == 1:
            # In case, it's only a 1D array, allways return the whole
            # array
            idx = slice(None, None, None)
        else:
            # return only the slice
            idx = (slice(None, None, None), ) + key

        return self.idx_obj[idx]

    def __setitem__(self, key, value):

        if not isinstance(key, tuple):
            # If only one index is handed over, convert key to tuple
            key = (key, )

        if np.ndim(self.idx_obj) == 1:
            # In case, it's only a 1D array, allways return the whole
            # array
            idx = slice(None, None, None)
        else:
            # return only the slice
            idx = (slice(None, None, None), ) + key

        self.idx_obj[idx] = value
        return self.idx_obj
