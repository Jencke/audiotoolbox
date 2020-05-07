import numpy as np
import audiotools as audio

class FrequencyDomainSignal(object):
    def __init__(self):
        self.waveform = np.array([], dtype=np.complex128)
        self.wav = self.waveform
        self.__freq = np.array([], dtype=np.complex128)
        self.__fs = None
        self.__is_norm = False #decides if the signal is normalized

    def from_timedomain(self, signal):
        fs = signal.fs
        wv = np.fft.fft(signal.waveform, axis=0)
        self.__freq = np.fft.fftfreq(len(wv), 1. / fs)
        self.set_waveform(wv, fs)

        return self

    @property
    def freq(self):
        """Get the signals sampling rate"""

        return self.__freq

    @property
    def fs(self):
        """Get the signals sampling rate"""

        return self.__fs
    @fs.setter
    def fs(self, fs):
        """Set the signals sampling rate"""

        # If no fs provided or allready defined:
        if fs == None and self.__fs == None:
            raise ValueError('No sampling rate provided')

        # If fs is defined
        elif fs != None:
            if self.__fs == None:
                self.__fs = fs
            elif self.__fs != fs:
                raise ValueError('Sampling rate can\'t be changed')
    @property
    def n_samples(self):
        """Get the number of samples in the signal"""
        if np.all(np.isnan(self.waveform)):
            return 0
        else:
            return self.waveform.shape[0]

    @property
    def real(self):
        return self.waveform.real

    @property
    def imag(self):
        return self.waveform.imag

    @property
    def phase(self):
        return np.angle(self.waveform)
    angle = phase

    @property
    def mag(self):
        return self.abs()

    def abs(self):
        return np.abs(self.waveform)

    def copy(self):
        """Returns a deepcopy of the signal"""
        return copy.deepcopy(self)

    def __getitem__(self, key):
        sig = FrequencyDomainSignal()
        sig.set_waveform(self.waveform[:, key], self.fs)

        return sig

    def to_timedomain(self):
        wv = np.fft.ifft(self.waveform, axis=0)
        wv = np.real_if_close(wv)
        signal = audio.Signal().set_waveform(wv, self.fs)
        return signal

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
        else:
            self.waveform[:] = waveform

        return self
