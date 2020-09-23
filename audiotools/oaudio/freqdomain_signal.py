import numpy as np
import audiotools as audio
from copy import deepcopy

from .base_signal import BaseSignal

class FrequencyDomainSignal(BaseSignal):
    # def __new__(cls, n_channels, duration, fs, dtype=complex):
    #     obj = BaseSignal.__new__(cls, n_channels, duration, fs, dtype)
    #     obj._is_norm = False #decides if the signal is normalized
    #     obj._freq = np.array([])
    #     return obj

    def __init__(self, *args, **kwargs):
        self._is_norm = False #decides if the signal is normalized
        self._freq = np.array([])

    def from_timedomain(self, signal, normalize=True):

        self[:] = np.fft.fft(signal, axis=0)
        if normalize:
            self /= signal.n_samples
            self._is_norm = True
        self._freq = np.fft.fftfreq(self.n_samples, 1. / self.fs)

        return self

    @property
    def freq(self):
        """Returns the frequencies"""
        return self._freq

    @property
    def omega(self):
        """Returns the circular frequencies"""
        return self._freq * 2 * np.pi

    @property
    def phase(self):
        """Retruns the phase of the waveform"""
        return np.angle(self)
    angle = phase

    # @property
    # def mag(self):
    #     """Retruns the absolute value of the waveform"""
    #     return self.abs()

    # def abs(self):
    #     """Retruns the absolute value of the waveform"""
    #     return np.abs(self.waveform)

    # def normalize(self):
    #     """Normalize the waveform by dividing by the number of samples

    #     This Normalize the Waveform by deviding all coefficents by the number
    #     of samples. The Normalized state is saved internaly so that
    #     normalization has no influence on reverse transformation. When
    #     converted from the timedomain, all coefficents are allready normalized
    #     by default

    #     Returns:
    #     --------
    #     FrequencyDomainSignal : Returns itself

    #     """
    #     if not self._is_norm:
    #         self.waveform /= self.n_samples
    #         self._is_norm = True
    #     return self

    # def phase_shift(self, phase):
    #     shift_val = 1.0j * phase * np.sign(self.freq)
    #     print (shift_val)
    #     self[:] = self * np.exp(shift_val)
    #     return self

    def to_timedomain(self):
        """Convert to timedomain.

        Convert to timedomain by means of inverse DFT. If the complex part
        after DFT is small (< 222e-16), it is neglected.

        Returns:
        --------
        Signal : The timedomain representation

        """
        if self._is_norm:
            self *= self.n_samples
        wv = np.fft.ifft(self, axis=0)
        wv = np.real_if_close(wv)
        signal = audio.Signal(self.n_channels, self.duration, self.fs)
        signal[:] = wv
        return signal
