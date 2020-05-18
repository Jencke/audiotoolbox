import numpy as np
import audiotools as audio
from audiotools.filter import brickwall, gammatone
import copy

class BaseSignal(object):
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
    waveform = np.array([])
    _fs = None

    # setter and getter to handle the sample rates
    @property
    def fs(self):
        """Get the signals sampling rate"""

        return self._fs
    @fs.setter
    def fs(self, fs):
        """Set the signals sampling rate"""

        # If no fs provided or allready defined:
        if fs == None and self._fs == None:
            raise ValueError('No sampling rate provided')

        # If fs is defined
        elif fs != None:
            if self._fs == None:
                self._fs = fs
            elif self._fs != fs:
                raise ValueError('Sampling rate can\'t be changed')

    # setter and getter to handle the number of channels in the signal
    @property
    def n_channels(self):
        """Get the number of channels in the signal"""
        if np.all(np.isnan(self.waveform)):
            return 0
        elif np.ndim(self.waveform) == 1:
            return 1
        else:
            return self.waveform.shape[1]

    @property
    def n_samples(self):
        """Get the number of samples in the signal"""
        if np.all(np.isnan(self.waveform)):
            return 0
        else:
            return self.waveform.shape[0]

    def copy(self, empty=False):
        """Returns a deepcopy of the signal"""
        cp_signal = copy.deepcopy(self)
        if empty:
            cp_signal.waveform[:, ...] = 0
        return cp_signal

    def add(self, value):
        """Add to the waveform

        Implement the addition of values, vectors or signals to the waveform.

        Parameters:
        -----------
        value : Signal, scalar, ndarray
            The value to add to waveform

        Returns:
        --------
        Signal : Returns itself

        """
        if self._size_matches(value):
            self.waveform += value.waveform
        elif (isinstance(value, int) or isinstance(value, float)
              or isinstance(value, np.ndarray)):
            self.waveform += value
        else:
            raise NotImplementedError('Can\'t add type %s to signal' % type(value))

        return self

    def subtract(self, value):
        """Subtract from the waveform

        Implement the subtraction of values, vectors or signals to the waveform.

        Parameters:
        -----------
        value : Signal, scalar, ndarray
            The value to subtract from waveform

        Returns:
        --------
        Signal : Returns itself

        """
        if self._size_matches(value):
            self.waveform -= value.waveform
        elif (isinstance(value, int) or isinstance(value, float)
              or isinstance(value, np.ndarray)):
            self.waveform -= value
        else:
            raise NotImplementedError('Can\'t add type %s to signal' % type(value))

        return self

    def multiply(self, value):
        """Multiply to the waveform

        Implement the multiplicaion of values, vectors or signals to the waveform.

        Parameters:
        -----------
        value : Signal, scalar, ndarray
            The value to multiply to waveform

        Returns:
        --------
        Signal : Returns itself

        """
        if self._size_matches(value):
            self.waveform *= value.waveform
        elif (isinstance(value, int) or isinstance(value, float)
              or isinstance(value, np.ndarray)):
            self.waveform *= value
        else:
            raise NotImplementedError('Can\'t add type %s to signal' % type(value))

        return self

    def divide(self, value):
        """Divide the waveform

        Implement the division of the waveform by values, vectors or signals to the.

        Parameters:
        -----------
        value : Signal, scalar, ndarray
            The value to divide the waveform with

        Returns:
        --------
        Signal : Returns itself

        """
        if self._size_matches(value):
            self.waveform /= value.waveform
        elif (isinstance(value, int) or isinstance(value, float)
              or isinstance(value, np.ndarray)):
            self.waveform /= value
        else:
            raise NotImplementedError('Can\'t add type %s to signal' % type(value))

        return self

    def append(self, signal):

        if isinstance(signal, Signal):
            new_wv = np.concatenate([self.waveform, signal.waveform])
        self.waveform = new_wv

        return self

    def __getitem__(self, key):
        sig = self.copy()
        sig.waveform = self.waveform[:, key]

        return sig

    def __add__(self, value):
        new_sig = self.copy()
        new_sig.add(value)
        return new_sig

    def __sub__(self, value):
        new_sig = self.copy()
        new_sig.subtract(value)
        return new_sig

    def __mul__(self, value):
        new_sig = self.copy()
        new_sig.multiply(value)
        return new_sig

    def __truediv__(self, value):
        new_sig = self.copy()
        new_sig.divide(value)
        return new_sig

    def _size_matches(self, signal):
        istype = isinstance(signal, type(self))
        if istype:
            isfs = signal.fs == self.fs
            isshape = signal.waveform.shape == self.waveform.shape
        else:
            return istype
        return istype and isfs and isshape

    @property
    def duration(self):
        """Get the duration of the signal in seconds"""
        if np.all(np.isnan(self.waveform)):
            return 0
        else:
            duration = self.n_samples / self.fs

            return duration


    def init_signal(self, n_channels, duration, fs, dtype=np.float64):
        """Initialize a signal with zeros

        Use this function to initialize a signal with zeros.  This
        also overwrites the current waveform.

        Parameters:
        -----------
        n_channels : int
            number of channels
        duration : float
            Signal duration in seconds
        fs : int
            sampling rate

        Returns:
        --------
        Signal : Returns itself

        """
        n_samples = audio.nsamples(duration, fs)
        if n_channels == 1:
            self.waveform = np.zeros([n_samples], dtype=dtype)
        else:
            self.waveform = np.zeros([n_samples, n_channels], dtype=dtype)
        self._fs = fs

        return self

    def set_waveform(self, waveform, fs=None):
        """Set the Waveform

        This function is a save method to set the waveform of a signal
        it should be prefered over directly setting the
        signal.waveform attribute.  if `waveform` does not match the
        current waveform in number of channels and samples, it is
        necessary to re-initialize the waveform using Signal.init_signal


        Parameters:
        -----------
        waveform : ndarray, dtype=complex
            The new waveform. shape must fit current signal
        fs : None or scalar
            Use only if no no samplingrate has ben set yet. Otherwise use None (default)

        """
        assert isinstance(waveform, np.ndarray)

        self.fs = fs

        # If the waveform was not previously initialized
        if self.waveform.shape[0] == 0:
            self.waveform = waveform
            self._freq = np.fft.fftfreq(self.n_samples, 1. / fs)
        else:
            self.waveform[:] = waveform

        return self
