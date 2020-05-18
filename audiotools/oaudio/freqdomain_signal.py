import numpy as np
import audiotools as audio
from copy import deepcopy

from .base_signal import BaseSignal

class FrequencyDomainSignal(BaseSignal):
    _is_norm = False #decides if the signal is normalized
    _freq = np.array([])

    def __init__(self):
        self.waveform = self.waveform.astype(np.complex128)

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
        """Get the frequencies"""
        return self._freq

    @property
    def omega(self):
        """Get the circular frequencies"""
        return self._freq * 2 * np.pi

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

    def normalize(self):
        if not self._is_norm:
            self.waveform /= self.n_samples
            self._is_norm = True
        return self

    def abs(self):
        return np.abs(self.waveform)

    def to_timedomain(self):
        if self._is_norm:
            self.waveform *= self.n_samples
        wv = np.fft.ifft(self.waveform, axis=0)
        wv = np.real_if_close(wv)
        signal = audio.Signal().set_waveform(wv, self.fs)
        return signal
