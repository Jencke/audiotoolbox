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
        """Get the time assigned to each sample of the signal"""
        time = audio.get_time(self, self.fs)
        return time

    def add_tone(self, frequency, amplitude=1, start_phase=0):
        r"""Add a sine tone with a given frequency, amplitude and start_phase

        This function will add a pure tone to the current
        waveform. following the equation:
        .. math:: x = x + cos(2\pi f t + \phi_0)
        where x is the waveform, f is the frequency, t is the time and
        phi_0 the starting phase.  The first evulated timepoint is 0.

        Parameters:
        -----------
        frequency : scalar
            The tone frequency in Hz.
        amplitude : scalar, optional
            The amplitude of the cosine. (default = 1)
        start_phase : scalar, optional
            The starting phase of the cosine. (default = 0)

        Returns:
        --------
        Signal : Returns itself

        """
        wv = audio.generate_tone(frequency,
                                 self.duration,
                                 self.fs,
                                 start_phase)

        # If multiple channels are defined, stack them.
        if self.n_channels > 1:
            wv = np.tile(wv, [self.n_channels, 1]).T
        self += amplitude * wv

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

    def add_noise(self, ntype='white', seed=None):
        r"""Add uncorrelated noise to the signal

        Add uncorrelated noise of a given spectral shape to all
        channels of the signal. Possible spectral shapes are 'white',
        'pink' (1 / f) and 'brown' (1 / f^2).


        Parameters:
        -----------
        ntype : {'white', 'pink', 'brown'}
            spectral shape of the noise
        seed : int or 1-d array_like, optional
            Seed for `RandomState`.
            Must be convertible to 32 bit unsigned integers.

        Returns:
        --------
        Signal : Returns itself

        """
        noise = audio.generate_noise(self.duration, self.fs,
                                     ntype=ntype, n_channels=1,
                                     seed=seed)

        if self.n_channels > 1:
            self += noise[:, None]
        else:
            self += noise

        return self

    # def add_corr_noise(self, corr=1, channels=[0, 1], seed=None):

    #     noise = audio.generate_corr_noise(self.duration, self.fs, corr, seed=seed)
    #     for i_c, n_c in enumerate(channels):
    #         summed_wv = self[n_c].waveform + noise[:, i_c]
    #         self[n_c].set_waveform(summed_wv)

    #     return self

    def set_dbspl(self, dbspl):
        """Set sound pressure level in dB

        Normalizes the signal to a given sound pressure level in dB
        relative 20e-6 Pa.

        Parameters:
        -----------
        dbspl : float
            The sound pressure level in dB

        Returns:
        --------
        Signal : Returns itself

        """

        res = audio.set_dbspl(self, dbspl)
        self[:] = res[:]

        return self

    def set_dbfs(self, dbfs):
        """Normalize the signal to a given dBFS RMS value.

        Parameters:
        -----------
        dbfs : float
            The dBFS RMS value

        Returns:
        --------
        Signal : Returns itself

        """

        nwv = audio.set_dbfs(self, dbfs)
        self[:] = nwv

        return self

    def calc_dbfs(self):
        """Calculate the dBFS RMS value for the signal

        Returns:
        --------
        float : The dBFS RMS value

        """
        dbfs = audio.calc_dbfs(self)
        return dbfs

    def calc_crest_factor(self):
        """Calculate crest factor

        Calculates the crest factor of the input signal. The crest factor
            is defined as:

        .. math:: C = \frac{|x_{peak}|}{x_{rms}}

        where :math:`x_{peak}` is the maximum of the absolute value and
        :math:`x{rms}` is the effective value of the signal.

        Returns:
        --------
        scalar :
            The crest factor

        """
        crest_factor = audio.crest_factor(self)
        return crest_factor


    # def bandpass(self, f_center, bw, ftype):
    #     if ftype == 'brickwall':
    #         f_low = f_center - 0.5 * bw
    #         f_high = f_center + 0.5 * bw
    #         filt_signal = brickwall(self.waveform, self.fs, f_low, f_high)
    #     elif ftype == 'gammatone':
    #         # f_low = f_center - 0.5 * bw
    #         # f_high = f_center + 0.5 * bw
    #         filt_signal = gammatone(self.waveform, self.fs, f_center, bw).real
    #     else:
    #         raise NotImplementedError('Filter type %s not implemented' % ftype)

    #     self.set_waveform(filt_signal)

    #     return self

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
        """Calculate the sound pressure level of the signal

        Returns:
        --------
        float : The sound pressure level in dB

        """
        dbspl = audio.calc_dbspl(self)
        return dbspl

    def zeropad(self, number=None, duration=None):
        """Add zeros to start and end of signal

        This function adds zeros of a given number or duration to the start or
        end of a signal. The same number of zeros is added to the start and
        end of a signal if a scalar is given as `number` or `duration. If a
        vector of two values is given, the first defines the number at the
        beginning, the second the number of zeros at the end.

        Parameters:
        -----------
        number : scalar or vecor of len(2), optional
            Number of zeros.
        duration : scalar or vecor of len(2), optional
            duration of zeros in seconds.

        Returns:
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
        """Add a fade in/out window to the signal

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

        Parameters:
        -----------
        frequency : float
          The frequency of the cosine modulator.
        m : float, optional
          The modulation index. (Default = 1)
        start_phase : float
          The starting phase of the cosine in radiant.

        Returns:
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

    def delay(self, delay, channels, method='fft'):

        to_delay = self[:, channels]
        if method == 'sample':
            nshift = int(np.round(nshift))
            shifted = audio.shift_signal(to_delay, nshift, mode='cyclic')
        elif method == 'fft':
            shifted = audio.fftshift_signal(to_delay, delay, self.fs)

        self.waveform[:, channels] = shifted

    #     return self


    def phase_shift(self, phase):
        """Shifts all frequency components of a signal by a constant phase.

        Shift all frequency components of a given signal by a constant
        phase by means of fFT transformation, phase shifting and inverse
        transformation.

        Parameters:
        -----------
        signal : ndarray
            The input signal
        phase : scalar
            The phase in rad by which the signal is shifted.

        Returns:
        --------
        ndarray :
            The phase shifted signal

        """
        wv = audio.phase_shift(self, phase, self.fs)
        self[:] = wv

        return self

    # def from_wav(self, filename, fullscale=True):
    #     wv, fs = wav.readwav(filename, fullscale)

    #     if wv.ndim > 1:
    #         n_channels = wv.shape[1]
    #     else:
    #         n_channels = 1

    #     duration = wv.shape[0] / fs
    #     self.init_signal(n_channels, duration, fs)
    #     self.set_waveform(wv)

    # def play(self, bitdepth=32, buffsize=1024):
    #     wv = self.waveform
    #     audio.interfaces.play(signal=wv,
    #                           fs=self.fs,
    #                           bitdepth=bitdepth,
    #                           buffsize=buffsize)

    # def plot(self, ax=None):
    #     import matplotlib.pyplot as plt
    #     if not ax:
    #         fig, ax = plt.subplots(1, 1)
    #     else:
    #         fig = ax.figure
    #     if self.n_channels == 2:
    #         ax.plot(self.time, self[0].waveform, color=audio.COLOR_L)
    #         ax.plot(self.time, self[1].waveform, color=audio.COLOR_R)
    #     else:
    #         ax.plot(self.time, self.waveform)
    #     return fig, ax

    def rms(self, axis=0):
        """Root mean square for each channel

        """

        rms = np.sqrt(np.mean(self**2, axis=axis))
        return rms

    def amplitude_spectrum(self, single_sided=False, nfft=None):
        """Amplitude spectrum of the signal

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
        """One-way rectification of the signal"""
        self.waveform[self.waveform < 0] = 0
        return self


    def to_freqdomain(self):
        fd = audio.oaudio.FrequencyDomainSignal(self.n_channels,
                                                self.duration, self.fs,
                                                dtype=complex)
        fd.from_timedomain(self)

        return fd

    # def to_analytical(self):
    #     an = audio.oaudio.AnalyticalSignal()
    #     an.from_timedomain(self)

    #     return an

    # def __repr__(self):
    #     repr = "Signal(channels={channels}, samples={samples}, fs={fs} Hz, duration={duration} s)".format(channels=self.n_channels, fs=self.fs, duration=self.duration, samples=self.n_samples)
    #     return repr
