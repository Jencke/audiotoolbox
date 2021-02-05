import numpy as np
import audiotools as audio
from audiotools.filter import brickwall, gammatone
from audiotools import wav
from .base_signal import BaseSignal
import copy

class Signal(BaseSignal):
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
        Returns itself : Signal

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
        Returns itself : Signal

        See Also
        --------
        audiotools.generate_noise
        audiotools.generate_uncorr_noise
        audiotools.Signal.add_uncorr_noise
        """
        noise = audio.generate_noise(self.duration, self.fs,
                                     ntype=ntype, n_channels=1,
                                     seed=seed)

        self[:] = (self.T + noise.T * np.sqrt(variance)).T
        return self

    def add_uncorr_noise(self, corr=0, variance=1, seed=None):
        r"""Add partly uncorrelated noise

        This function adds partly uncorrelated noise using the N+1
        generator method.

        To generate N partly uncorrelated noises with a desired
        correlation coefficent of $\rho$, the algoritm first generates N+1
        noise tokens which are then orthogonalized using the Gram-Schmidt
        process (as implementd in numpy.linalg.qr). The N+1 th noise token
        is then mixed with the remaining noise tokens using the equation

        .. math:: X_{\rho,n} = X_{N+1}  \sqrt{\rho} + X_n  \beta \sqrt{1 - \rho}

        where :math:`X_{\rho,n}` is the nth output and noise,
        :math:`X_{n}` the nth indipendent noise and :math:`X_{N=1}` is the
        common noise.

        for two noise tokens, this is identical to the assymetric
        three-generator method described in [1]_

        Parameters
        ----------
        corr : int, optional
            Desired correlation of the noise tokens, (default=0)
        variance : scalar, optional
            The desired variance of the noise, (default=1)
        seed : int or 1-d array_like, optional
            Seed for `RandomState`.
            Must be convertible to 32 bit unsigned integers.

        Returns
        -------
        Returns itself : Signal

        See Also
        --------
        audiotools.generate_noise
        audiotools.generate_uncorr_noise
        audiotools.Signal.add_noise

        References
        ----------

        .. [1] Hartmann, W. M., & Cho, Y. J. (2011). Generating partially
          correlated noise—a comparison of methods. The Journal of the
          Acoustical Society of America, 130(1),
          292–301. http://dx.doi.org/10.1121/1.3596475

        """

        noise = audio.generate_uncorr_noise(duration=self.duration,
                                            fs=self.fs,
                                            n_channels=self.n_channels,
                                            corr=corr,
                                            seed=seed)

        self += noise * np.sqrt(variance)

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
        Returns itself : Signal

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
        Returns itself : Signal

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

        Calculates the crest factor for the signal. The crest factor
        is defined as:

        .. math:: C = \frac{|x_{peak}|}{x_{rms}}

        where :math:`x_{peak}` is the maximum of the absolute value and
        :math:`x_{rms}` is the effective value of the signal.

        Returns
        --------
        scalar :
            The crest factor

        See Also
        --------
        audiotools.crest_factor

        """
        crest_factor = audio.crest_factor(self)
        return crest_factor


    def bandpass(self, f_center, bw, ftype, **kwargs):
        r"""Apply a bandpass filter

        Applies a bandpass filter to the signal. The availible filters are:

        - brickwall: A 'optimal' brickwall filter
        - gammatone: A real valued gammatone filter

        For additional filter parameters and detailed description see
        the respective implementations:

        - :meth:`audiotools.filter.brickwall`
        - :meth:`audiotools.filter.gammatone`

        Parameters
        ----------
        f_center : scalar
            The banddpass center frequency in Hz
        bw : scalar
            The filter bandwidth in Hz
        ftype : {'brickwall', 'gammatone'}
            The filtertype
        **kwargs :
            Further keyword arguments are passed to the respective filter functions

        Returns
        --------
            Returns itself : Signal

        See Also
        --------
        audiotools.filter.brickwall
        audiotools.filter.gammatone

        """

        if ftype == 'brickwall':
            f_low = f_center - 0.5 * bw
            f_high = f_center + 0.5 * bw
            filt_signal = brickwall(self, self.fs, f_low, f_high)
        elif ftype == 'gammatone':
            if 'return_complex' not in kwargs:
                kwargs['return_complex'] = False
            filt_signal = gammatone(self, self.fs, f_center, bw, **kwargs)
        else:
            raise NotImplementedError('Filter type %s not implemented' % ftype)

        # in case of complex output, signal needs to be reshaped and
        # typecast
        if np.iscomplexobj(filt_signal):
            shape = self.shape
            self.dtype = complex
            self.resize(shape, refcheck=False)
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
        end of a signal.

        If number or duration is a scalar, an equal number of zeros
        will be appended at the front and end of the array. If a
        vector of two values is given, the first defines the number or duration at
        the beginning, the second the number or duration of zeros at the end.

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
        audiotools.zeropad
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

        return self

    def add_fade_window(self, rise_time, type='cos', **kwargs):
        r"""Add a fade in/out window to the signal

        This function multiplies a fade window with a given rise time
        onto the signal. for mor information about the indiviual
        window functions refer to the implementations:

        - cos: A rasied cosine window :meth:`audiotools.cosine_fade_window`
        - gauss: A gaussian window :meth:`audiotools.gaussian_fade_window`


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
        audiotools.gaussian_fade_window
        audiotools.cosine_fade_window

        """

        if type == 'gauss':
            win = audio.gaussian_fade_window(self, rise_time,
                                             self.fs, **kwargs)
        elif type == 'cos':
            win = audio.cosine_fade_window(self, rise_time,
                                           self.fs, **kwargs)
        self *= win
        return self

    def add_cos_modulator(self, frequency, m, start_phase=0):
        r"""Multiply a cosinus amplitude modulator to the signal

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
        audiotools.cos_amp_modulator

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
        inverse transform. This is identical to using the
        :meth:'audiotools.FrequencyDomainSignal.time_shift'
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

        if method == 'sample':
            nshift = audio.nsamples(delay, self.fs)
            shifted = audio.shift_signal(self, nshift)
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
        Signal :
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

    def rms(self):
        r"""Root mean square

        Returns
        -------
        float : The RMS value
        """

        rms = np.sqrt(np.mean(self**2))
        return rms

    # def amplitude_spectrum(self, single_sided=False, nfft=None):
    #     r"""Amplitude spectrum of the signal

    #     """

    #     nfft = nfft if nfft else self.n_samples
    #     spec = np.fft.fft(self, n=nfft, axis=0) / nfft
    #     freq = np.fft.fftfreq(nfft, 1.0 / self.fs)
    #     spec = np.fft.fftshift(spec, axes=0)
    #     freq = np.fft.fftshift(freq, axes= 0)

    #     if single_sided:
    #         freq = freq[nfft // 2:, ...]
    #         spec = spec[nfft // 2:, ...]
    #         spec *= 2
    #         spec[0, ...] /= 2 # do not double dc
    #         if not nfft % 2:
    #             spec[-1, ...] /= 2       # nyquist bin should also not be doubled

    #     return freq, spec

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
        r"""One-way rectification of the signal

        Returns
        -------
        Returns itself : Signal

        """
        self[self < 0] = 0
        return self

    def writewav(self, filename, bitdepth=16):
        wav.writewav(filename, self, self.fs, bitdepth)

    def to_freqdomain(self):
        r"""Convert to frequency domain by applying a DFT

        This function returns a frequency domain representation of the
        signal.

        As opposed to most methods, this conversion is not in-place
        but a new :meth:'audiotools.FrequencyDomainSignal' object is
        returned

        Returns
        -------
        The frequency domain representation of the signal : FrequencyDomainSignal

        """
        fd = audio.oaudio.FrequencyDomainSignal(self.n_channels,
                                                self.duration, self.fs,
                                                dtype=complex)
        fd.from_timedomain(self)

        return fd

    def to_analytical(self):
        r"""Convert to analytical signal representation

        This function converts the signal into its analytical
        representation. The function is not applied inplace but a new
        signal with datatype complex is returned

        Returns
        -------
        The analytical signal : Signal

        """
        fd_signal = self.to_freqdomain()
        a_signal = fd_signal.to_analytical().to_timedomain()
        return a_signal
