"""Signal mixins for organizing Signal class functionality."""

import signal
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
from scipy.signal import get_window

import resampy

import warnings
from ... import audiotoolbox as audio

if TYPE_CHECKING:
    from ..signal import Signal


class ModificationMixin:
    """Mixin for signal modification methods."""

    def set_dbspl(self, dbspl: float):
        r"""Set sound pressure level in dB.

        Normalizes the signal to a given sound pressure level in dB
        relative 20e-6 Pa.

        Normalizes the signal to a given sound pressure level in dB
        relative 20e-6 Pa.
        for this, the Signal is multiplied with the factor :math:`A`

        .. math:: A = \frac{p_0}{\sigma} 10^{L / 20}

        where :math:`L` is the goal SPL, :math:`p_0=20\mu Pa` and
        :math:`\sigma` is the RMS of the signal.


        Parameters
        ----------
        dbspl : float
            The sound pressure level in dB

        Returns
        -------
        Returns itself : Signal

        """
        p0 = 20e-6  # ref_value

        factor = (p0 * 10 ** (float(dbspl) / 20)) / self.stats.rms

        self *= factor

        return self

    def set_dbpeak(self, dbpeak: float):
        """Peak normalization of the signal.

        Normalizes the signal in relation to it's peak amplitude. 0dB peak corresponds to a maximum amplitude of 1.

        Parameters:
        -----------
        dbpeak : float
            The peak dB value to reach

        Returns
        -------
        Returns itself : Signal
        """
        peak_val = np.max(np.abs(self), axis=0)
        factor = (10 ** (float(dbpeak) / 20)) / peak_val
        self *= factor

        return self

    def set_dbfs(self, dbfs: float):
        r"""Full scale normalization of the signal.

        Normalizes the signal Level to dB Fullscale. 0dB FS corresponds to
        a signal with an rms of :math:`\frac{1}{\sqrt{2}}` so that a tone at 0dBS will
        have an amplitude of 1.

        Parameters
        ----------
        dbfs : float
            The db full scale value to reach

        Returns
        -------
        self: Signal
        """

        rms0 = 1 / np.sqrt(2)

        factor = (rms0 * 10 ** (float(dbfs) / 20)) / self.stats.rms
        # elif norm == "peak":
        #     peak_val = np.max(self, axis=0)
        #     factor = (10 ** (float(dbfs) / 20)) / peak_val

        # else:
        #     raise (ValueError('norm must be "rms" or "peak"'))
        self *= factor
        return self

    def add_fade_window(self, rise_time: float, win_type: str = "hann", **kwargs):
        r"""Add a fade in/out window to the signal.

        This function multiplies a fade window with a given rise time
        onto the signal.


        Parameters
        ----------
        rise_time : float
            The rise time in seconds.
        win_type : str
            Any window function supported by scipy.signal.get_window. Default is 'hann'.
        **kwargs
            Additional keyword arguments passed to the window function (see scipy implementation).

        Notes
        -----
        Window types:

        - boxcar
        - triang
        - blackman
        - hamming
        - hann
        - bartlett
        - flattop
        - parzen
        - bohman
        - blackmanharris
        - nuttall
        - barthann
        - cosine
        - exponential
        - tukey
        - taylor
        - lanczos
        - kaiser (needs beta)
        - kaiser_bessel_derived` (needs beta)
        - gaussian` (needs standard deviation)
        - general_cosine` (needs weighting coefficients)
        - general_gaussian` (needs power, width)
        - general_hamming` (needs window coefficient)
        - dpss` (needs normalized half-bandwidth)
        - chebwin` (needs attenuation)

        Returns
        -------
        Return itself : Signal

        """

        # for compatibility, cos equals a raised cosine
        if win_type == "cos":
            win_type = "hann"

        n_samples = audio.nsamples(
            rise_time, fs=self.fs
        )  # calculate number of samples for fade window

        # The full window needs to be twice the size
        win = get_window(win_type, 2 * n_samples, **kwargs)[:n_samples]

        # Empty signal for storing the fade window
        fade_win = audio.Signal(1, self.duration, self.fs)
        fade_win[:] = 1

        # Multiply the first half of the window with the beginning and end of the window
        fade_win[:n_samples] = win
        fade_win[-n_samples:] = win[::-1]

        # Reshape the fade window to match the signal
        new_shape = (self.n_samples,) + (1,) * (self.ndim - 1)
        fade_win = fade_win.reshape(new_shape)

        self *= fade_win
        return self

    def add_cos_modulator(self, frequency: float, m: float, start_phase: float = 0):
        r"""Multiply a cosinus amplitude modulator to the signal.

        Multiplies a cosinus amplitude modulator following the equation:

        .. math:: 1 + m  \cos{2  \pi  f_m  t  \phi_{0}}

        where :math:`m` is the modulation depth, :math:`f_m` is the
        modualtion frequency and :math:`t` is the time. :math:`\phi_0` is the
        start phase

        Parameters
        -----------
        frequency : float
            The frequency of the cosine modulator.
        m : float, optional
            The modulation index. (Default = 1)
        start_phase : float
            The starting phase of the cosine in radiant.

        Returns
        --------
        Returns itself : Signal

        See Also
        --------
        audiotoolbox.cos_amp_modulator

        """

        modulator = 1 + m * np.cos(2 * np.pi * frequency * self.time + start_phase)

        self *= modulator
        return self

    def delay(self, delay: float, method: Literal["fft", "sample"] = "fft"):
        r"""Delays the signal by circular shifting.

        Circular shift the functions foreward to create a certain time
        delay relative to the orginal time. E.g if shifted by an
        equivalent of N samples, the value at sample i will move to
        sample i + N.

        Two methods can be used. Using the default method 'fft', the
        signal is shifted by applyint a FFT transform, and phase
        shifting each frequency accoring to the delay and applying an
        inverse transform. This is identical to using the
        :meth:'audiotoolbox.FrequencyDomainSignal.time_shift'
        method. When using the method 'sample', the signal is time
        delayed by circular shifting the signal by the number of
        samples that is closest to delay.

        Parameters
        -----------
        delay : float
            The delay in secons
        method : {'fft', 'samples'} optional
            The method used to delay the signal (default: 'fft')

        Returns
        --------
        Signal :
            Returns itself

        See Also
        --------
        audio.shift_signal
        audio.FreqDomainSignal.time_shift

        """
        if method == "sample":
            nshift = audio.nsamples(delay, self.fs)
            shifted = audio.shift_signal(self, nshift)
        elif method == "fft":
            shifted = self.to_freqdomain().time_shift(delay).to_timedomain()

        self[:] = shifted
        return self

    def phase_shift(self, phase: float):
        r"""Shifts all frequency components of a signal by a constant phase.

        Shift all frequency components of a given signal by a constant
        phase. This is identical to calling the phase_shift method of
        the FrequencyDomainSignal class.

        Parameters
        -----------
        phase : scalar
            The phase in rad by which the signal is shifted.

        Returns
        --------
        Signal :
            Returns itself

        """
        wv = self.to_freqdomain().phase_shift(phase).to_timedomain()
        self[:] = wv

        return self

    def trim(self, t_start: float, t_end: Union[float, None] = None):
        r"""Trim the signal between two points in time.

        removes the number of samples according to t_start and
        t_end. This method can not be applied to a single channel or
        slice.

        Parameters
        -----------
        t_start: float
            Signal time at which the returned signal should start
        t_end: float or None (optional)
            Signal time at which the signal should stop. The full remaining
            signal is used if set to None. (default: None)

        Returns
        --------
        Signal :
            Returns itself
        """
        if not isinstance(self.base, type(None)):
            raise RuntimeError("Trimming can not be applied to slices")

        # calculate the indices at which the signal should be trimmed
        i_start = audio.nsamples(t_start, self.fs)
        if t_end:
            if t_end < 0:
                t_end = self.duration + t_end
            i_end = audio.nsamples(t_end, self.fs)
        else:
            i_end = self.n_samples

        #  store the cliped part in the signal
        self[0 : i_end - i_start] = self[i_start:i_end]

        newshape = list(self.shape)
        newshape[0] = i_end - i_start
        self.resize(newshape, refcheck=False)

        return self

    def zeropad(
        self,
        number: Union[None, tuple[int, int]] = None,
        duration: Union[None, tuple[float, float]] = None,
    ):
        r"""Add zeros to start and end of signal.

        This function adds zeros of a given number or duration to the start or
        end of a signal.

        If number or duration is a scalar, an equal number of zeros
        will be appended at the front and end of the array. If a
        vector of two values is given, the first defines the number or
        duration at the beginning, the second the number or duration
        of zeros at the end.

        Parameters
        -----------
        number : scalar or vecor of len(2), optional
            Number of zeros.
        duration : scalar or vecor of len(2), optional
            duration of zeros in seconds.

        Returns
        --------
        Returns itself : Signal

        """
        # Only one number or duration must be stated
        if duration is None and number is None:
            raise ValueError("Must state duration or number of zeros")
        elif duration is None and number is None:
            raise ValueError("Must state only duration or number of zeros")
            return

        # If duration instead of number is stated, calculate the
        # number of samples to buffer with
        elif duration is not None and number is None:
            if not np.isscalar(duration):
                n_s = audio.nsamples(duration[0], self.fs)
                n_e = audio.nsamples(duration[1], self.fs)
            else:
                n_s = n_e = audio.nsamples(duration, self.fs)
        else:
            if not np.isscalar(number):
                n_s = number[0]
                n_e = number[1]
            else:
                n_s = n_e = number

        # Can only be applied to the whole signal not to a slice
        if not isinstance(self.base, type(None)):
            raise RuntimeError("Zeropad can only be applied to" " the whole signal")
        else:
            orig_nsamp = self.n_samples
            new_shape = (orig_nsamp + n_s + n_e,) + self.shape[1:]
            self.resize(new_shape, refcheck=False)
            self[n_s : n_s + orig_nsamp] = self[:orig_nsamp]
            self[:n_s] = 0
            self[-n_e:] = 0
        return self

    def rectify(self):
        r"""One-way rectification of the signal.

        Returns
        -------
        Returns itself : Signal

        """
        self[self < 0] = 0
        return self

    def apply_gain(self, gain: float):
        r"""Applys gain factor to the signal

        Fixed gain by multiplying the signal with a fixed factor calculated as

        .. math:: 10^{(G / 20)}

        where G is the gain.

        Parameters:
        -----------
        gain : float
            The gain factor in dB

        Returns
        -------
        Returns itself : Signal

        """
        mult_fac = 10 ** (gain / 20)
        self *= mult_fac

        return self

    def resample(self, new_fs: int):
        """Resample the signal to a new sampling rate.

        This method uses the `resampy` library to resample the signal to a new
        sampling rate. It is based on the band-limited sinc interpolation method
        for sampling rate conversion as described by Smith (2015). [1]_.

        .. [1] Smith, Julius O. Digital Audio Resampling Home Page
            Center for Computer Research in Music and Acoustics (CCRMA),
            Stanford University, 2015-02-23.
            Web published at `<http://ccrma.stanford.edu/~jos/resample/>`_.
        """

        if new_fs <= 0 and not isinstance(new_fs, int):
            raise ValueError("new_fs must be a positive integer")
        if not isinstance(self.base, type(None)):
            raise RuntimeError("Zeropad can only be applied to" " the whole signal")
        else:
            out = resampy.resample(x=self, sr_orig=self.fs, sr_new=new_fs, axis=0)
            self.resize(out.shape, refcheck=False)
            self[:] = out
            self._fs = new_fs
            return self
