import numpy as np
import audiotools as audio
from audiotools.filter import brickwall
from audiotools import wav
import copy


class Signal(object):
    r"""
    Attributes:
    -----------
    waveform : ndarray
      The signals waveform
    fs
    n_channels
    n_samples
    duration
    time
    """
    def __init__(self):
        self.waveform = np.array([])
        self.__fs = None

    # setter and getter to handle the sample rates
    @property
    def fs(self):
        """Get the signals sampling rate"""

        return self.__fs
    @fs.setter
    def fs(self, fs):
        """Set the signals sampling rate"""

        # If no fs provided or allready defined:
        if fs == None and self.__fs == None:
            raise ValueError('No sampling rate provided')

        # If fs is defined
        elif fs != None:
            if self.__fs == None:
                self.__fs = fs
            elif self.__fs != fs:
                raise ValueError('Sampling rate can\'t be changed')

    # setter and getter to handle the number of channels in the signal
    @property
    def n_channels(self):
        """Get the number of channels in the signal"""
        if np.all(np.isnan(self.waveform)):
            return 0
        elif np.ndim(self.waveform) == 1:
            return 1
        else:
            return self.waveform.shape[1]

    @property
    def n_samples(self):
        """Get the number of samples in the signal"""
        if np.all(np.isnan(self.waveform)):
            return 0
        else:
            return self.waveform.shape[0]

    @property
    def duration(self):
        """Get the duration of the signal in seconds"""
        if np.all(np.isnan(self.waveform)):
            return 0
        else:
            duration = self.n_samples / self.fs

            return duration

    @property
    def time(self):
        """Get the time assigned to each sample of the signal"""
        time = audio.get_time(self.waveform, self.fs)
        return time

    def set_waveform(self, waveform, fs=None):
        """Set the Waveform

        This function is a save method to set the waveform of a signal
        it should be prefered over directly setting the
        signal.waveform attribute.  if `waveform` does not match the
        current waveform in number of channels and samples, it is
        necessary to re-initialize the waveform using Signal.init_signal


        Parameters:
        -----------
        waveform : ndarray
            The new waveform. shape must fit current signal
        fs : None or scalar
            Use only if no no samplingrate has ben set yet. Otherwise use None (default)

        """
        assert isinstance(waveform, np.ndarray)

        self.fs = fs

        # If the waveform was not previously initialized
        if self.waveform.shape[0] == 0:
            self.waveform = waveform
        else:
            self.waveform[:] = waveform

    def init_signal(self, n_channels, duration, fs):
        """Initialize a signal with zeros

        Use this function to initialize a signal with zeros.  This
        also overwrites the current waveform.

        Parameters:
        -----------
        n_channels : int
            number of channels
        duration : float
            Signal duration in seconds
        fs : int
            sampling rate

        Returns:
        --------
        Signal : Returns itself

        """
        n_samples = audio.nsamples(duration, fs)
        if n_channels == 1:
            self.waveform = np.zeros([n_samples])
        else:
            self.waveform = np.zeros([n_samples, n_channels])
        self.__fs = fs

        return self

    def copy(self):
        """Returns a deepcopy of the signal"""
        return copy.deepcopy(self)


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
        summed_wv = self.waveform + amplitude * wv
        self.set_waveform(summed_wv)

        return self

    def add_noise(self, seed=None):
        noise = np.random.randn(self.n_samples)
        if self.n_channels > 1:
            noise = np.tile(noise, [self.n_channels, 1]).T
        summed_wv = self.waveform + noise
        self.set_waveform(summed_wv)

        return self

    def add_corr_noise(self, corr=1, channels=[0, 1], seed=None):
        noise = audio.generate_corr_noise(self.duration, self.fs, corr, seed=seed)
        for i_c, n_c in enumerate(channels):
            summed_wv = self[n_c].waveform + noise[:, i_c]
            self[n_c].set_waveform(summed_wv)

        return self

    def set_dbspl(self, dbspl):
        """Normalize the signal to a given sound pressure level in dB.

        Parameters:
        -----------
        dbspl : float
            The sound pressure level in dB

        Returns:
        --------
        Signal : Returns itself

        """

        nwv = audio.set_dbspl(self.waveform, dbspl)
        self.set_waveform(nwv)

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

        nwv = audio.set_dbfs(self.waveform, dbfs)
        self.set_waveform(nwv)

        return self

    def calc_dbfs(self):
        """Calculate the dBFS RMS value for the signal

        Returns:
        --------
        float : The dBFS RMS value

        """
        dbfs = audio.calc_dbfs(self.waveform)
        return dbfs

    def bandpass(self, f_center, bw, ftype):
        if ftype == 'brickwall':
            f_low = f_center - 0.5 * bw
            f_high = f_center + 0.5 * bw
            filt_signal = brickwall(self.waveform, self.fs, f_low, f_high)
        else:
            raise NotImplementedError('Filter type %s not implemented' % type(value))

        self.set_waveform(filt_signal)

        return self

    def calc_dbspl(self):
        """Calculate the sound pressure level of the signal

        Returns:
        --------
        float : The sound pressure level in dB

        """
        dbspl = audio.calc_dbspl(self.waveform)
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

        wv = audio.zeropad(self.waveform, number)
        self.waveform = wv
        return self

    def fade_window(self, rise_time, type='cos'):
        return self.add_fade_window(rise_time, type)
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
            win = audio.gaussian_fade_window(self.waveform, rise_time,
                                             self.fs)
        elif type == 'cos':
            win = audio.cosine_fade_window(self.waveform, rise_time,
                                           self.fs)
        elif type == 'hann':
            win = audio.hann_fade_window(self.waveform, rise_time,
                                         self.fs)

        wv = self.waveform * win
        self.set_waveform(wv)
        return self


    def add(self, value):
        """Add to the waveform

        Implement the addition of values, vectors or signals to the waveform.

        Parameters:
        -----------
        value : Signal, scalar, ndarray
            The value to add to waveform

        Returns:
        --------
        Signal : Returns itself

        """
        if self.__size_matches(value):
            self.waveform += value.waveform
        elif (isinstance(value, int) or isinstance(value, float)
              or isinstance(value, np.ndarray)):
            self.waveform += value
        else:
            raise NotImplementedError('Can\'t add type %s to signal' % type(value))

        return self

    def subtract(self, value):
        """Subtract from the waveform

        Implement the subtraction of values, vectors or signals to the waveform.

        Parameters:
        -----------
        value : Signal, scalar, ndarray
            The value to subtract from waveform

        Returns:
        --------
        Signal : Returns itself

        """
        if self.__size_matches(value):
            self.waveform -= value.waveform
        elif (isinstance(value, int) or isinstance(value, float)
              or isinstance(value, np.ndarray)):
            self.waveform -= value
        else:
            raise NotImplementedError('Can\'t add type %s to signal' % type(value))

        return self

    def multiply(self, value):
        """Multiply to the waveform

        Implement the multiplicaion of values, vectors or signals to the waveform.

        Parameters:
        -----------
        value : Signal, scalar, ndarray
            The value to multiply to waveform

        Returns:
        --------
        Signal : Returns itself

        """
        if self.__size_matches(value):
            self.waveform *= value.waveform
        elif (isinstance(value, int) or isinstance(value, float)
              or isinstance(value, np.ndarray)):
            self.waveform *= value
        else:
            raise NotImplementedError('Can\'t add type %s to signal' % type(value))

        return self

    def divide(self, value):
        """Divide the waveform

        Implement the division of the waveform by values, vectors or signals to the.

        Parameters:
        -----------
        value : Signal, scalar, ndarray
            The value to divide the waveform with

        Returns:
        --------
        Signal : Returns itself

        """
        if self.__size_matches(value):
            self.waveform /= value.waveform
        elif (isinstance(value, int) or isinstance(value, float)
              or isinstance(value, np.ndarray)):
            self.waveform /= value
        else:
            raise NotImplementedError('Can\'t add type %s to signal' % type(value))

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

        mod = audio.cos_amp_modulator(signal=self.waveform,
                                      modulator_freq=frequency,
                                      fs=self.fs,
                                      mod_index=m)

        wv = self.waveform * mod
        self.set_waveform(wv)
        return self

    def delay(self, delay, channels, method='fft', mode='zeros'):

        nshift = delay * self.fs

        if delay == 0: return self

        # Allways use the sample algorithm if the delay is a full
        # multiple of the time resolution
        if nshift % 1 == 0:
            method = 'sample'

        to_delay = self.waveform[:, channels]
        if method == 'sample':
            nshift = int(np.round(nshift))
            shifted = audio.shift_signal(to_delay, nshift, mode)
        elif method == 'fft':
            shifted = audio.fftshift_signal(to_delay, delay, self.fs, mode)

        if mode =='cyclic':
            self.waveform[:, channels] = shifted
        else:
            self.waveform[:, channels] = shifted[:self.n_samples, :]

        return self


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
        wv = audio.phase_shift(self.waveform, phase, self.fs)
        self.set_waveform(wv)

        return self

    def from_wav(self, filename, fullscale=True):
        wv, fs = wav.readwav(filename, fullscale)

        if wv.ndim > 1:
            n_channels = wv.shape[1]
        else:
            n_channels = 1

        duration = wv.shape[0] / fs
        self.init_signal(n_channels, duration, fs)
        self.set_waveform(wv)

    def play(self, bitdepth=32, buffsize=1024):
        wv = self.waveform
        audio.interfaces.play(signal=wv,
                              fs=self.fs,
                              bitdepth=bitdepth,
                              buffsize=buffsize)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax:
            ax.plot(self.time, self.waveform)
        else:
            fig, ax = plt.subplots(1, 1)
            ax.plot(self.time, self.waveform)
        return fig, ax

    def mean(self, axis=0):
        mean = self.waveform.mean(axis=axis)
        return mean

    def rms(self, axis=0):
        rms = np.sqrt(np.mean(self.waveform**2, axis=axis))
        return rms

    def append(self, signal):
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

        if isinstance(signal, Signal):
            new_wv = np.concatenate([self.waveform, signal.waveform])
        self.waveform = new_wv

        return self

    def __repr__(self):
        repr = "Signal(channels={channels}, samples={samples}, fs={fs} Hz, duration={duration} s)".format(channels=self.n_channels, fs=self.fs, duration=self.duration, samples=self.n_samples)
        return repr

    def __getitem__(self, key):
        sig = Signal()
        sig.set_waveform(self.waveform[:, key], self.fs)

        return sig

    def __add__(self, value):
        new_sig = self.copy()
        new_sig.add(value)
        return new_sig

    def __sub__(self, value):
        new_sig = self.copy()
        new_sig.subtract(value)
        return new_sig

    def __mul__(self, value):
        new_sig = self.copy()
        new_sig.multiply(value)
        return new_sig

    def __truediv__(self, value):
        new_sig = self.copy()
        new_sig.divide(value)
        return new_sig

    def __size_matches(self, signal):
        istype = isinstance(signal, Signal)
        if istype:
            isfs = signal.fs == self.fs
            isshape = signal.waveform.shape == self.waveform.shape
        else:
            return istype
        return istype and isfs and isshape
