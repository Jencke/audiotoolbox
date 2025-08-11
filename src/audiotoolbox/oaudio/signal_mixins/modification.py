"""Signal mixins for organizing Signal class functionality."""

from typing import TYPE_CHECKING

import numpy as np

import resampy

import warnings
from ... import audiotoolbox as audio

if TYPE_CHECKING:
    from ..signal import Signal


class ModificationMixin:
    """Mixin for signal modification methods."""

    def set_dbspl(self, dbspl):
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

        See Also
        --------
        audiotoolbox.set_dbspl
        audiotoolbox.Signal.calc_dbspl
        audiotoolbox.Signal.set_dbfs
        audiotoolbox.Signal.calc_dbfs

        """
        res = audio.set_dbspl(self, dbspl)
        self[:] = res[:]

        return self

    def set_dbfs(self, dbfs):
        r"""Normalize the signal to a given dBFS RMS value.

        Normalizes the signal to dB Fullscale
        for this, the Signal is multiplied with the factor :math:`A`

        .. math:: A = \frac{1}{\sqrt{2}\sigma} 10^\frac{L}{20}

        where :math:`L` is the goal Level, and :math:`\sigma` is the
        RMS of the signal.

        Parameters
        ----------
        dbfs : float
            The dBFS RMS value in dB

        Returns
        -------
        Returns itself : Signal

        Examples
        --------
        >>> sig = Signal(1, 1, 48000).add_tone(1000)
        >>> sig.set_dbfs(-3)
        >>> sig.stats.dbfs
        -3.0



        See Also
        --------
        audiotoolbox.set_dbspl
        audiotoolbox.set_dbfs
        audiotoolbox.calc_dbfs
        audiotoolbox.Signal.set_dbspl
        audiotoolbox.Signal.calc_dbspl
        audiotoolbox.Signal.calc_dbfs

        """
        nwv = audio.set_dbfs(self, dbfs)
        self[:] = nwv

        return self

    def add_fade_window(self, rise_time, type="cos", **kwargs):
        r"""Add a fade in/out window to the signal.

        This function multiplies a fade window with a given rise time
        onto the signal. for mor information about the indiviual
        window functions refer to the implementations:

        - cos: A rasied cosine window :meth:`audiotoolbox.cosine_fade_window`
        - gauss: A gaussian window :meth:`audiotoolbox.gaussian_fade_window`


        Parameters
        ----------
        rise_time : float
            The rise time in seconds.
        type : 'cos', 'gauss', 'cos2'
            The type of the window. (default = 'cos')

        Returns
        -------
        Return itself : Signal

        See Also
        --------
        audiotoolbox.gaussian_fade_window
        audiotoolbox.cosine_fade_window

        """
        if type == "gauss":
            win = audio.gaussian_fade_window(self, rise_time, self.fs, **kwargs)
        elif type == "cos":
            win = audio.cosine_fade_window(self, rise_time, self.fs, **kwargs)
        self *= win
        return self

    def add_cos_modulator(self, frequency, m, start_phase=0):
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
        mod = audio.cos_amp_modulator(
            duration=self,
            modulator_freq=frequency,
            fs=self.fs,
            mod_index=m,
            start_phase=start_phase,
        )
        self *= mod
        return self

    def delay(self, delay, method="fft"):
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

    def phase_shift(self, phase):
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

    def trim(self, t_start, t_end=None):
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

    def zeropad(self, number=None, duration=None):
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

        See Also
        --------
        audiotoolbox.zeropad

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
                number_s = audio.nsamples(duration[0], self.fs)
                number_e = audio.nsamples(duration[1], self.fs)
                number = (number_s, number_e)
            else:
                number = audio.nsamples(duration, self.fs)

        # Can only be applied to the whole signal not to a slice
        if not isinstance(self.base, type(None)):
            raise RuntimeError("Zeropad can only be applied to" " the whole signal")
        else:
            wv = audio.zeropad(self, number)
            self.resize(wv.shape, refcheck=False)
            self[:] = wv

        return self

    def rectify(self):
        r"""One-way rectification of the signal.

        Returns
        -------
        Returns itself : Signal

        """
        self[self < 0] = 0
        return self

    def apply_gain(self, gain):
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
