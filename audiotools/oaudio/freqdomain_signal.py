import numpy as np
import audiotools as audio
from copy import deepcopy

from .base_signal import BaseSignal

class FrequencyDomainSignal(BaseSignal):

    def from_timedomain(self, signal):
        self[:] = np.fft.fft(signal, axis=0)
        self /= signal.n_samples

        return self


    @property
    def freq(self):
        """Returns the frequencies"""
        freq = np.fft.fftfreq(self.n_samples, 1. / self.fs)
        return freq

    @property
    def omega(self):
        """Returns the circular frequencies"""
        return self.freq * 2 * np.pi

    @property
    def phase(self):
        """Retruns the phase of the waveform"""
        return np.angle(self)
    angle = phase

    @property
    def mag(self):
        """Retruns the absolute value of the waveform"""
        return self.abs()

    # def abs(self):
    #     """Retruns the absolute value of the waveform"""
    #     return np.abs(self.waveform)

    # def timeshift(self, time):
    #     phases = audio.time2phase(time, self.freq)
    #     self.waveform *= np.exp(1j * phases)
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

        # revert normalization
        self *= self.n_samples
        wv = np.fft.ifft(self, axis=0)
        wv = np.real_if_close(wv)
        signal = audio.Signal(self.n_channels, self.duration, self.fs)
        signal[:] = wv
        return signal
