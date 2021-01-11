import numpy as np
import audiotools as audio
from copy import deepcopy
from scipy.signal import hilbert

from .base_signal import BaseSignal

class AnalyticalSignal(BaseSignal):

    def __new__(cls, n_channels, duration, fs, dtype=complex):
        obj = BaseSignal.__new__(cls, n_channels, duration, fs, dtype)
        return obj

    def to_freqdomain(self):
        signal = audio.Signal(self.n_channels, self.duration, self.fs)

    def to_timedomain(self):
        signal = audio.Signal(self.n_channels, self.duration, self.fs)
        signal[:] = np.real(self)
        return signal

    # def from_freqdomain(self, signal):
    #     signal = signal.copy()
    #     nsamp = signal.n_samples

    #     # Xf = sp_fft.fft(x, N, axis=axis)
    #     h = np.zeros(nsamp)

    #     if nsamp % 2 == 0:
    #         h[0] = h[nsamp // 2] = 1
    #         h[1:nsamp // 2] = 2
    #     else:
    #         h[0] = 1
    #         h[1:(nsamp + 1) // 2] = 2

    #     # if x.ndim > 1:
    #     #     ind = [np.newaxis] * x.ndim
    #     #     ind[axis] = slice(None)
    #     #     h = h[tuple(ind)]
    #     fdmoain_sig *= h
    #     x = sp_fft.ifft(Xf * h, axis=axis)


    #     self[:] = np.fft.fft(signal, axis=0)
    #     self /= signal.n_samples
    #     return self


    # def to_timedomain(self):
    #     """Convert to timedomain.

    #     Convert to timedomain by means of inverse DFT. If the complex part
    #     after DFT is small (< 222e-16), it is neglected.

    #     Returns:
    #     --------
    #     Signal : The timedomain representation

    #     """

    #     # revert normalization
    #     self *= self.n_samples
    #     wv = np.fft.ifft(self, axis=0)
    #     wv = np.real_if_close(wv)
    #     signal = audio.Signal(self.n_channels, self.duration, self.fs)
    #     signal[:] = wv
    #     return signal
