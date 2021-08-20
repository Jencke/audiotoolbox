import numpy as np
import audiotools as audio

from . import base_signal


def _copy_to_dim(array, dim):
    if np.ndim(dim) == 0:
        dim = (dim,)

    # tile by the number of dimensions
    tiled_array = np.tile(array, (*dim[::-1], 1)).T
    # squeeze to remove axis of lenght 1
    tiled_array = np.squeeze(tiled_array)

    return tiled_array


class FrequencyDomainSignal(base_signal.BaseSignal):
    """Base class for signals in the frequency domain.

    Parameters
    ----------
    n_channels : int or tuple
      Number of channels to be used, can be N-dimensional
    duration : float
      Stimulus duration in seconds
    fs : int
      Sampling rate  in Hz
    dtype : complex, optional
      Datatype of the array (default is float)

    Returns
    -------
    Signal : The new object.

    """

    def __new__(cls, n_channels, duration, fs, dtype=complex):
        obj = base_signal.BaseSignal.__new__(cls, n_channels,
                                             duration, fs, dtype)
        return obj

    @property
    def freq(self):
        r"""Return the frequency axis

        Returns
        -------
        The frequency axis in Hz : numpy.ndarray

        """
        freq = np.fft.fftfreq(self.n_samples, 1. / self.fs)
        return freq

    @property
    def omega(self):
        r"""Return the angular frequency axis

        Returns
        -------
        The angular frequencies : numpy.ndarray

        """
        return self.freq * 2 * np.pi

    @property
    def phase(self):
        r"""The Argument of the frequency components

        Returns
        -------
        The arguments in radiants : numpy.ndarray

        """
        return np.angle(self)
    angle = phase

    @property
    def mag(self):
        r"""Returns the magnitudes of the frequency components

        equivalent to :meth:`audiotools.FrequencyDomainSignal.abs()`

        Returns
        -------
        The magnitudes : numpy.ndarray
        """
        return self.abs()

    def time_shift(self, time):
        r"""Apply a time shift

        Time shift the signal by a linear phase shift by
        multiplying all frequency components with:

        .. math:: H(\omega) = e^{-j\omega\Delta t}

        where :math:`j` is the complex unit, :math:`omega` the
        circular frequency and :math:`\Delta t` the timeshift.

        Parameters
        ----------
        time : scalar
            The time by which the signal should be shifted

        Returns
        -------
        Returns itself : FrequencyDomainSignal

        Also See
        --------
        audiotools.Signal.delay

        """

        phases = - self.omega * time

        # fix the last bin in case of odd samples in order to keep the
        # tranformed signal real
        if not self.n_samples % 2:
            phases[self.n_samples//2] = 0

        shift_factor = np.exp(1j * phases)
        shift_factor = _copy_to_dim(shift_factor, self.shape[1:])

        self *= shift_factor

        return self

    def phase_shift(self, phase):
        r"""Apply a phase shift

        Phase shift the spectrum by multiplying all frequency
        components with:

        .. math:: H(\omega) = e^{-j\psi}

        where :math:`j` is the complex unit and :math:`\psi` is the
        desired phaseshift.

        Parameters
        ----------
        phase : scalar
            The phaseshift

        Returns
        -------
        Returns itself : FrequencyDomainSignal

        Also See
        --------
        audiotools.Signal.phase_shift

        """

        shift_val = - 1.0j * phase * np.sign(self.freq)
        shift_val = _copy_to_dim(shift_val, self.shape[1:])

        self *= np.exp(shift_val)
        return self


    def from_timedomain(self, signal):
        self[:] = np.fft.fft(signal, axis=0)
        self /= signal.n_samples
        return self

    def to_timedomain(self):
        """Convert to timedomain.

        Convert to timedomain by means of inverse DFT. If the complex
        part after DFT is small (< 222e-16), it is neglected. This
        method is not applied in-place but a new
        :meth:'audiotools.Signal' object is returned

        Returns:
        --------
        The timedomain representation: Signal

        """

        # revert normalization
        self *= self.n_samples
        wv = np.fft.ifft(self, axis=0)
        wv = np.real_if_close(wv)
        signal = audio.Signal(self.n_channels, self.duration, self.fs, dtype = wv.dtype)
        signal[:] = wv
        return signal

    def to_analytical(self):
        """Convert spectrum to analytical signal

        Converts the spectrum to that of the equivalent analytical
        signal by removing the negative frequency components and
        doubling the positive coponents.

        Returns:
        --------
        Returns itself : FrequencyDomainSignal

        """
        nsamp = self.n_samples

        h = np.zeros(nsamp)

        if nsamp % 2 == 0:
            # Do not change the nyquist bin or the offset
            h[0] = h[nsamp // 2] = 1
            # double all other components
            h[1:nsamp // 2] = 2
        else:
            h[0] = 1
            h[1:(nsamp + 1) // 2] = 2

        if self.ndim > 1:
            ind = [np.newaxis] * self.ndim
            ind[0] = slice(None)
            h = h[tuple(ind)]

        self *= h

        return self
