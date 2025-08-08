"""Definition for the Signal class."""

from typing import Type, cast, Union, Literal

import numpy as np

from . import base_signal
from .. import audiotoolbox as audio
from .freqdomain_signal import FrequencyDomainSignal
from .stats import SignalStats
from .time_frequency import TimeFrequency
from scipy.signal import fftconvolve
import warnings

# Import all mixins
from .signal_mixins import (
    AnalysisMixin,
    GenerationMixin,
    ModificationMixin,
    IOMixin,
    FilteringMixin,
)


class Signal(
    base_signal.BaseSignal,
    AnalysisMixin,
    GenerationMixin,
    ModificationMixin,
    IOMixin,
    FilteringMixin,
):
    """Base class for signals in the timedomain.

    Parameters
    ----------
    n_channels : int or tuple
        Number of channels to be used, can be N-dimensional
    duration : float
        Stimulus duration in seconds
    fs : int
        Sampling rate  in Hz
    dtype : type, optional
        Datatype of the array (default is float)

    Returns
    -------
    Signal : The new signal object.

    Examples
    --------
    Create a 1 second long signal with two channels at a sampling rate
    of 48 kHz

    >>> sig = audiotoolbox.Signal(2, 1, 48000)
    >>> print(sig.shape)
    (4800, 2)

    """

    def __new__(
        cls: Type[base_signal.BaseSignal],
        n_channels: Union[int, tuple, list],
        duration: float,
        fs: int,
        dtype=float,
    ):
        """Create new objects."""
        obj = super().__new__(cls, n_channels, duration, fs, dtype)
        obj.stats = SignalStats(obj)
        obj.time_frequency = TimeFrequency(obj)
        return cast(Signal, obj)

    def __array_finalize__(self, obj):
        """Finalyze signal."""
        # Finalize Array __new__ is only called when directly
        # creating a new object.  When copying or templating, __new__ is
        # not called which is why init code should be put in
        # __array_finalize__

        base_signal.BaseSignal.__array_finalize__(self, obj)

        if obj is None:
            # When creating new array
            self.time_offset = 0
        else:
            # When copying or slicing
            self.time_offset = getattr(obj, "time_offset", None)
            self.stats = SignalStats(self)
            self.time_frequency = TimeFrequency(self)

        return obj

    @property
    def time(self):
        r"""Time vector for the signal."""
        time = audio.get_time(self, self.fs) + self.time_offset
        return time

    def plot(self, ax=None):
        """Plot the Signal using matplotlib.

        This function quickly plots the signal over time. If the
        signal only contains two channels, they are plotted in blue
        and red.

        Currently only works for signals with 1 dimensional channel
        shape.

        Parameters
        ----------
        ax : None, matplotlib.axis (optional)
            The axis that should be used for plotting. If None, a new
            figure is created. (default is None)

        """
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

    def to_freqdomain(self):
        r"""Convert to frequency domain by applying a DFT.

        This function returns a frequency domain representation of the
        signal.

        As opposed to most methods, this conversion is not in-place
        but a new :meth:`audiotoolbox.FrequencyDomainSignal` object is
        returned

        Returns
        -------
        FrequencyDomainSignal :
            The frequency domain representation of the signal

        """
        fd = FrequencyDomainSignal(
            self.n_channels, self.duration, self.fs, dtype=complex
        )
        fd.from_timedomain(self)

        return fd

    def to_analytical(self):
        r"""Convert to analytical signal representation.

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


def as_signal(signal, fs):
    """Convert Numpy array to Signal class.

    Parameters
    ----------
    signal : ndarray
        The input array
    fs : int
        The sampling rate in Hz

    Returns
    -------
    The converted signal : Signal

    """
    # if allready signal class
    if isinstance(signal, Signal):
        return signal
    else:
        duration = len(signal) / fs
        if np.ndim(signal) == 0:
            n_channels = 1
        else:
            n_channels = signal.shape[1:]

        sig_out = Signal(n_channels, duration, fs, dtype=signal.dtype)
        sig_out[:] = signal
    return sig_out


__all__ = ["Signal", "as_signal"]
