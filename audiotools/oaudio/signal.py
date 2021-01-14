import numpy as np
import audiotools as audio
from audiotools.filter import brickwall, gammatone
from audiotools import wav
from .base_signal import BaseSignal
import copy

class Signal(BaseSignal):
    r"""
    Attributes:
    -----------
    fs
    n_channels
    n_samples
    duration
    time
    """
    @property
    def time(self):
        r"""The time vector for the signal"""
        time = audio.get_time(self, self.fs)
        return time

    def add_tone(self, frequency, amplitude=1, start_phase=0):
        r"""Add a cosine to the signal

        This function will add a pure tone to the current
        waveform. following the equation:

        .. math:: x = x + cos(2\pi f t + \phi_0)

        where :math:`x` is the waveform, :math:`f` is the frequency,
        :math:`t` is the time and :math:`\phi_0` the starting phase.
        The first evulated timepoint is 0.

        Parameters
        ----------
        frequency : scalar
            The tone frequency in Hz.
        amplitude : scalar, optional
            The amplitude of the cosine. (default = 1)
        start_phase : scalar, optional
            The starting phase of the cosine. (default = 0)

        Returns
        -------
        Returns itself

        See Also
        --------
        audiotools.generate_tone

        """
        wv = audio.generate_tone(frequency,
                                 self.duration,
                                 self.fs,
                                 start_phase)

        # If multiple channels are defined, stack them.
        # if self.n_channels > 1:
        #     wv = np.tile(wv, [self.n_channels, 1]).T
        self[:] = (self.T + amplitude * wv.T).T

        return self

    def add_low_noise_noise(self, low_f, high_f, n_rep=10, seed=None):
        noise = audio.generate_low_noise_noise(duration=self.duration,
                                               fs=self.fs,
                                               low_f=low_f,
                                               high_f=high_f,
                                               n_rep=n_rep,
                                               seed=seed)
        if self.n_channels > 1:
            summed_wv = self + noise[:, None]
        else:
            summed_wv = self + noise

        self[:] = summed_wv

        return self

    def add_noise(self, ntype='white', variance=1, seed=None):
        r"""Add uncorrelated noise to the signal

        add gaussian noise with a defined variance and different
        spectral shapes. The noise is generated in the frequency domain
        using the gaussian pseudorandom generator ``numpy.random.randn``.
        The real and imaginary part of each frequency component is set
        using the psudorandom generator. Each frequency bin is then
        weighted dependent on the spectral shape. The resulting spektrum
        is then transformed into the time domain using ``numpy.fft.ifft``

        Weighting functions:

         - white: :math:`w(f) = 1`
         - pink: :math:`w(f) = \frac{1}{\sqrt{f}}`
         - brown: :math:`w(f) = \frac{1}{f}`

        Parameters
        ----------
        ntype : {'white', 'pink', 'brown'}
            spectral shape of the noise
        variance : scalar, optional
            The Variance of the noise
        seed : int or 1-d array_like, optional
            Seed for `RandomState`.
            Must be convertible to 32 bit unsigned integers.

        Returns
        -------
        Returns itself

        See Also
        --------
        audiotools.generate_noise
        """
        noise = audio.generate_noise(self.duration, self.fs,
                                     ntype=ntype, n_channels=1,
                                     seed=seed)

        self[:] = (self.T + noise.T * np.sqrt(variance)).T
        return self

    def add_corr_noise(self, corr=1, channels=[0, 1], seed=None):

        noise = audio.generate_corr_noise(self.duration, self.fs, corr, seed=seed)
        for i_c, n_c in enumerate(channels):
            self[:, n_c] += noise[:, i_c]
            # summed_wv = self[n_c].waveform + noise[:, i_c]
            # self[n_c].set_waveform(summed_wv)

        return self

    def set_dbspl(self, dbspl):
        r"""Set sound pressure level in dB

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
        Returns itself

        See Also
        --------
        audiotools.set_dbspl
        audiotools.Signal.calc_dbspl
        audiotools.Signal.set_dbfs
        audiotools.Signal.calc_dbfs
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
        Returns itself

        See Also
        --------
        audiotools.set_dbspl
        audiotools.set_dbfs
        audiotools.calc_dbfs
        audiotools.Signal.set_dbspl
        audiotools.Signal.calc_dbspl
        audiotools.Signal.calc_dbfs

        """

        nwv = audio.set_dbfs(self, dbfs)
        self[:] = nwv

        return self

    def calc_dbfs(self):
        r"""Calculate the dBFS RMS value for the signal

        .. math:: L = 20 \log_10\left(\sqrt{2}\sigma\right)

        where :math:`\sigma` is the signals RMS.

        Returns
        -------
        float : The dBFS RMS value

        """
        dbfs = audio.calc_dbfs(self)
        return dbfs

    def calc_crest_factor(self):
        r"""Calculate crest factor

        Calculates the crest factor of the input signal. The crest factor
        is defined as:

        .. math:: C = \frac{|x_{peak}|}{x_{rms}}

        where :math:`x_{peak}` is the maximum of the absolute value and
        :math:`x{rms}` is the effective value of the signal.

        Returns
        --------
        scalar :
            The crest factor

        """
        crest_factor = audio.crest_factor(self)
        return crest_factor


    def bandpass(self, f_center, bw, ftype):
        r"""Apply a bandpass filter

        Applies a bandpass filter. The availible filters are:
        - brickwall: A 'optimal' brickwall filter
        - gammatone: A real valued gammatone filter

        .. math:: C = \frac{|x_{peak}|}{x_{rms}}

        where :math:`x_{peak}` is the maximum of the absolute value and
        :math:`x{rms}` is the effective value of the signal.

        Returns
        --------
        scalar :
            The crest factor

        """

        if ftype == 'brickwall':
            f_low = f_center - 0.5 * bw
            f_high = f_center + 0.5 * bw
            filt_signal = brickwall(self, self.fs, f_low, f_high)
        elif ftype == 'gammatone':
            filt_signal = gammatone(self, self.fs, f_center, bw).real
        else:
            raise NotImplementedError('Filter type %s not implemented' % ftype)

        self[:] = filt_signal

        return self

    # def lowpass(self, f_cut, ftype):
    #     if ftype == 'brickwall':
    #         filt_signal = brickwall(self.waveform, self.fs, 0, f_cut)
    #     else:
    #         raise NotImplementedError('Filter type %s not implemented' % ftype)

    #     self.set_waveform(filt_signal)

    #     return self

    # def highpass(self, f_cut, ftype):
    #     if ftype == 'brickwall':
    #         filt_signal = brickwall(self.waveform, self.fs, f_cut, np.inf)
    #     else:
    #         raise NotImplementedError('Filter type %s not implemented' % ftype)

    #     self.set_waveform(filt_signal)

    #     return self

    def calc_dbspl(self):
        r"""Calculate the sound pressure level of the signal


        .. math:: L = 20  \log_{10}\left(\frac{\sigma}{p_o}\right)

        where :math:`L` is the SPL, :math:`p_0=20\mu Pa` and
        :math:`\sigma` is the RMS of the signal.

        Returns
        --------
        float : The sound pressure level in dB

        """
        dbspl = audio.calc_dbspl(self)
        return dbspl

    def zeropad(self, number=None, duration=None):
        r"""Add zeros to start and end of signal

        This function adds zeros of a given number or duration to the start or
        end of a signal. The same number of zeros is added to the start and
        end of a signal if a scalar is given as `number` or `duration. If a
        vector of two values is given, the first defines the number at the
        beginning, the second the number of zeros at the end.

        Parameters
        -----------
        number : scalar or vecor of len(2), optional
            Number of zeros.
        duration : scalar or vecor of len(2), optional
            duration of zeros in seconds.

        Returns
        --------
        Signal : Returns itself

        """

        #Only one number or duration must be stated
        if duration == None and number == None:
            raise ValueError('Must state duration or number of zeros')
        elif duration == None and number == None:
                raise ValueError('Must state only duration or number of zeros')
                return

        # If duration instead of number is stated, calculate the
        # number of samples to buffer with
        elif duration != None and number == None:
            if not np.isscalar(duration):
                number_s = audio.nsamples(duration[0], self.fs)
                number_e = audio.nsamples(duration[1], self.fs)
                number = (number_s, number_e)
            else:
                number = audio.nsamples(duration, self.fs)

        # Can only be applied to the whole signal not to a slice
        if not isinstance(self.base, type(None)):
            raise RuntimeError('Zeropad can only be applied to the whole signal')
        else:
            wv = audio.zeropad(self, number)
            self.resize(wv.shape, refcheck=False)
            self[:] = wv
            # print('no slice')
        # new_sig = Signal(self.n_channels, duration, self.fs)
        # new_sig[:] = wv
        # self = new_sig
        return self

    def add_fade_window(self, rise_time, type='cos'):
        r"""Add a fade in/out window to the signal

        This function multiplies a fade window with a given rise time
        onto the signal. Possible values for `type` are 'cos' for a
        cosine window, 'gauss' for a gaussian window, 'hann' for a
        hann window.

        Paramters:
        ----------
        rise_time : float
            The rise time in seconds.
        type : 'cos', 'gauss' or 'hann'
            The type of the window. (default = 'cos')

        """

        if type == 'gauss':
            win = audio.gaussian_fade_window(self, rise_time,
                                             self.fs)
        elif type == 'cos':
            win = audio.cosine_fade_window(self, rise_time,
                                           self.fs)
        elif type == 'cos2':
            win = audio.cossquare_fade_window(self, rise_time,
                                           self.fs)
        elif type == 'hann':
            win = audio.hann_fade_window(self, rise_time,
                                         self.fs)

        self *= win
        return self

    def add_cos_modulator(self, frequency, m, start_phase=0):
        r"""Multiply a cosinus amplitude modulator to the signal

        Multiplies a cosinus amplitude modulator following the equation:
        ..math:: 1 + m * \cos{2 * \pi * f_m * t + \phi_{0}}

        where m is the modulation depth, f_m is the modualtion frequency
        and t is the time. \phi_0 is the start phase

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
        Signal : Returns itself

        """

        mod = audio.cos_amp_modulator(signal=self,
                                      modulator_freq=frequency,
                                      fs=self.fs,
                                      mod_index=m,
                                      start_phase=start_phase)
        self *= mod
        return self

    def delay(self, delay, method='fft'):
        r"""Delays the signal by circular shifting

        Circular shift the functions foreward to create a certain time
        delay relative to the orginal time. E.g if shifted by an
        equivalent of N samples, the value at sample i will move to
        sample i + N.

        Two methods can be used. Using the default method 'fft', the
        signal is shifted by applyint a FFT transform, and phase
        shifting each frequency accoring to the delay and applying an
        inverse transform. This is identical to using the time_shift()
        method of the FrequencyDomainSignal class. When using the
        method 'sample', the signal is time delayed by circular
        shifting the signal by the number of samples that is closest
        to delay.

        Parameters
        -----------
        delay : float
            The delay in secons
        method : {'fft', 'samples'} optional
            The method used to delay the signal (default: 'fft')

        Returns
        --------
        Signal:
            Returns itself

        """

        if method == 'sample':
            nshift = audio.nsamples(delay, self.fs)
            shifted = audio.shift_signal(self, nshift, mode='cyclic')
        elif method == 'fft':
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

    def clip(self, t_start, t_end=None):
        r"""Clip the signal between two points in time

        removes the number of semples according to t_start and
        t_end. This method can not be applied to a single channel or
        slice.

        Parameters
        -----------
        t_start: float
            Signal time at which the returned signal should start
        t_end: flot or None (optional)
           Signal time at which the signal should stop. The full remaining
           signal is used if set to None. (default: None)

        Returns
        --------
        Signal:
            Returns itself
        """

        if not isinstance(self.base, type(None)):
            raise RuntimeError('Clipping can not be applied to slices')

        # calculate the indices at which the signal should be cliped
        i_start = audio.nsamples(t_start, self.fs)
        if t_end:
            if t_end < 0:
                t_end = self.duration + t_end
            i_end = audio.nsamples(t_end, self.fs)
        else:
            i_end = self.n_samples

        #  store the cliped part in the signal
        self[0:i_end-i_start, :] = self[i_start:i_end, :]


        newshape = list(self.shape)
        newshape[0] = i_end - i_start
        self.resize(newshape, refcheck=False)

        return self

    def play(self, bitdepth=32, buffsize=1024):
        wv = self
        audio.interfaces.play(signal=wv,
                              fs=self.fs,
                              bitdepth=bitdepth,
                              buffsize=buffsize)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if not ax:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.figure
        if self.n_channels == 2:
            ax.plot(self.time, self[:, 0], color=audio.COLOR_L)
            ax.plot(self.time, self[:, 1], color=audio.COLOR_R)
        else:
            ax.plot(self.time, self)
        return fig, ax

    def rms(self, axis=0):
        r"""Root mean square for each channel
        """

        rms = np.sqrt(np.mean(self**2, axis=axis))
        return rms

    def amplitude_spectrum(self, single_sided=False, nfft=None):
        r"""Amplitude spectrum of the signal

        """

        nfft = nfft if nfft else self.n_samples
        spec = np.fft.fft(self, n=nfft, axis=0) / nfft
        freq = np.fft.fftfreq(nfft, 1.0 / self.fs)
        spec = np.fft.fftshift(spec, axes=0)
        freq = np.fft.fftshift(freq, axes= 0)

        if single_sided:
            freq = freq[nfft // 2:, ...]
            spec = spec[nfft // 2:, ...]
            spec *= 2
            spec[0, ...] /= 2 # do not double dc
            if not nfft % 2:
                spec[-1, ...] /= 2       # nyquist bin should also not be doubled

        return freq, spec

    # def phase_spectrum(self, nfft=None):
    #     nfft = nfft if nfft else self.n_samples
    #     freq, spec = self.amplitude_spectrum(nfft)
    #     phase = np.angle(spec)
    #     phase = phase[..., nfft // 2:]
    #     freq = freq[..., nfft // 2:]
    #     return freq, phase

    # def autopower_spectrum(self, nfft=None):
    #     nfft = nfft if nfft else self.n_samples
    #     freq, spec = self.amplitude_spectrum(nfft)
    #     auto_spec = np.real_if_close(spec * spec.conj())

    #     return freq, auto_spec

    # def power_spectrum(self, nfft=None):
    #     nfft = nfft if nfft else self.n_samples
    #     freq, specsubtype = self.autopower_spectrum(nfft)
    #     freq = freq[nfft // 2:, ...]
    #     spec = spec[nfft // 2:, ...]
    #     spec *= 2
    #     spec[0, ...] /= 2 # do not double dc
    #     if not nfft % 2:
    #         spec[-1, ...] /= 2       # nyquist bin should also not be doubled
    #     return freq, spec

    def rectify(self):
        r"""One-way rectification of the signal"""
        self[self < 0] = 0
        return self

    def writewav(self, filename, bitdepth=16):
        wav.writewav(filename, self, self.fs, bitdepth)

    def to_freqdomain(self):
        fd = audio.oaudio.FrequencyDomainSignal(self.n_channels,
                                                self.duration, self.fs,
                                                dtype=complex)
        fd.from_timedomain(self)

        return fd

    def to_analytical(self):
        fd_signal = self.to_freqdomain()
        a_signal = fd_signal.to_analytical()
        return a_signal
