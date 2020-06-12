import numpy as np
import audiotools as audio
from copy import deepcopy

from .base_signal import BaseSignal

class FrequencyDomainSignal(BaseSignal):
    _is_norm = False #decides if the signal is normalized
    _freq = np.array([])

    # def __init__(self, n_channels=None, duration=None, fs=None):
    #     if bool(n_channels) & bool(duration) & bool(fs):
    #         self.init_signal(n_channels, duration, fs, dtype=np.complex128)

    def from_timedomain(self, signal, normalize=True):
        fs = signal.fs
        wv = np.fft.fft(signal.waveform, axis=0)
        if normalize:
            wv /= signal.n_samples
            self._is_norm = True
        self._freq = np.fft.fftfreq(len(wv), 1. / fs)
        self.set_waveform(wv, fs)

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
    def real(self):
        """Retruns the real part of the waveform"""
        return self.waveform.real

    @property
    def imag(self):
        """Retruns the imaginary part of the waveform"""
        return self.waveform.imag

    @property
    def phase(self):
        """Retruns the phase of the waveform"""
        return np.angle(self.waveform)
    angle = phase

    @property
    def mag(self):
        """Retruns the absolute value of the waveform"""
        return self.abs()

    def abs(self):
        """Retruns the absolute value of the waveform"""
        return np.abs(self.waveform)

    def normalize(self):
        """Normalize the waveform by dividing by the number of samples

        This Normalize the Waveform by deviding all coefficents by the number
        of samples. The Normalized state is saved internaly so that
        normalization has no influence on reverse transformation. When
        converted from the timedomain, all coefficents are allready normalized
        by default

        Returns:
        --------
        FrequencyDomainSignal : Returns itself

        """
        if not self._is_norm:
            self.waveform /= self.n_samples
            self._is_norm = True
        return self

    # def phase_shift(self, phase):
    #     shift_val = 1.0j * phase * np.sign(self.freq)
    #     print (shift_val)
    #     self.waveform *= np.exp(shift_val)
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
            self.waveform *= self.n_samples
        wv = np.fft.ifft(self.waveform, axis=0)
        wv = np.real_if_close(wv)
        signal = audio.Signal().set_waveform(wv, self.fs)
        return signal
