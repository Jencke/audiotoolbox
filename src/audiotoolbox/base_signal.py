from typing import Any, cast

import numpy as np
import audiotoolbox as audio


class BaseSignal(np.ndarray):
    r"""Basic Signal class inherited by all Signal representations"""

    def __new__(
        cls,
        n_channels: int | tuple,
        duration: float,
        fs: int,
        dtype: Any = float,
    ):

        n_samples = audio.nsamples(duration, fs)

        # Always keep an explicit channel axis so mono and multichannel
        # signals use a consistent memory layout.
        obj = super(BaseSignal, cls).__new__(
            cls, shape=(n_samples, *np.atleast_1d(n_channels)), dtype=dtype
        )
        obj._fs = fs
        obj.fill(0)

        return obj

    def __array_finalize__(self, obj):
        # If called explicitly, obj = None
        if obj is None:
            return

        # If it was called after e.g slicing, copy
        # copy sample rate
        self._fs = getattr(obj, "_fs", None)

    def __getitem__(self, key) -> Any:
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        try:
            return super().__setitem__(key, value)
        except ValueError:
            target = np.ndarray.__getitem__(self, key)
            arr = np.asarray(value)

            # Compatibility path: allow assigning a mono 1D vector (N,)
            # into explicit-channel slices like (N, 1).
            if (
                isinstance(target, np.ndarray)
                and target.ndim >= 2
                and target.shape[-1] == 1
                and arr.ndim == target.ndim - 1
                and arr.shape == target.shape[:-1]
            ):
                return super().__setitem__(key, arr[..., np.newaxis])

            # Compatibility path in the other direction: if a view resolves
            # to 1D (N,), accept incoming mono-column data shaped (N, 1).
            if (
                isinstance(target, np.ndarray)
                and target.ndim == 1
                and arr.ndim == 2
                and arr.shape[1] == 1
                and arr.shape[0] == target.shape[0]
            ):
                return super().__setitem__(key, arr[:, 0])
            raise

    @property  # getter to handle the sample rates
    def fs(self) -> int:
        """Sampling rate of the signal in Hz"""

        assert self._fs is not None
        return cast(int, self._fs)

    # getter to handle the number of channels in the signal
    @property
    def n_channels(self):
        """Number of channels in the signal"""
        if self.ndim == 1:
            return 1
        elif self.ndim == 2:
            return self.shape[1]
        else:
            return self.shape[1:]

    @property
    def channel_shape(self) -> tuple:
        """Tuple describing the channel axes shape."""
        if self.ndim == 1:
            return (1,)
        return self.shape[1:]

    @property
    def n_samples(self):
        """Number of samples in the signal"""
        return self.shape[0]

    @property
    def duration(self):
        """Duration of the signal in seconds"""
        duration = self.n_samples / self.fs

        return duration

    @property
    def ch(self):
        r"""Direct channel indexer

        Returns an indexer class which enables direct indexing and
        slicing of the channels independent of samples.

        Channel indices address only the trailing channel axes; the
        sample axis is always preserved as the leading axis. If a
        selection resolves to a single logical channel, the result is
        normalized to the canonical mono shape ``(n_samples, 1)``.

        Examples
        --------
        >>> sig = audiotoolbox.Signal((2, 3), 1, 48000).add_noise()
        >>> print(sig.ch[1, 2].shape)
        (48000, 1)
        >>> print(sig.ch[1].shape)
        (48000, 3)
        >>> print(sig.ch[:, 2].shape)
        (48000, 2)

        """
        return _chIndexer(self)

    def concatenate(self, signal):
        """Concatenate another signal or array

        This method appends another signal to the end of the current
        signal.

        Parameters
        -----------
        signal : signal or ndarray
            The signal to append

        Returns
        --------
        Returns itself

        """
        if not isinstance(self.base, type(None)):
            raise RuntimeError("Can only concatenate to a full signal")
        else:
            old_n = self.n_samples
            new_n = old_n + signal.n_samples
            new_shape = list(self.shape)
            new_shape[0] = new_n
            self.resize(new_shape, refcheck=False)
            self[old_n:] = signal
        return self

    def multiply(self, x: float | np.ndarray) -> "BaseSignal":
        """In-place multiplication

        This function allows for in-place multiplication

        Parameters
        -----------
        x : scalar or ndarray
            The value or array to multiply with the signal

        Returns
        --------
        Returns itself

        Examples
        --------
        >>> sig = audiotoolbox.Signal(1, 1, 48000).add_tone(500).multiply(2)
        >>> print(sig.max())
        2.0

        """
        self *= x
        return cast(BaseSignal, self)

    def add(self, x) -> "BaseSignal":
        """In-place summation

        This function allows for in-place summation.

        Parameters
        -----------
        x : scalar or ndarray
            The value or array to add to the signal

        Returns
        --------
        Returns itself

        Examples
        --------
        >>> sig = audiotoolbox.Signal(1, 1, 48000).add_tone(500).add(2)
        >>> print(sig.mean())
        2.0

        """

        self += x
        return cast(BaseSignal, self)

    def abs(self):
        """Absolute value

        Calculates the absolute value or modulus of all values of the signal

        """
        return np.abs(self)

    def copy_empty(self):
        out = self.copy()
        out[:] = 0
        return out

    def summary(self):
        if self.duration < 1:
            duration = f"{self.duration * 1000:2}ms"
        else:
            duration = f"{self.duration:2}s"

        if self.fs < 1000:
            fs = f"{self.fs}Hz"
        else:
            fs = f"{self.fs / 1000:.1f}kHz"

        samp = f"{self.n_samples} samples"

        chan = f"{self.n_channels} channel"

        repr = (
            duration
            + " @ "
            + fs
            + " = "
            + samp
            + " in "
            + chan
            + " | dtype: "
            + str(self.dtype)
        )
        return repr


class _chIndexer(object):
    """Channel Indexer

    Allows channels to be indexed directly without needing to care about
    samples

    """

    def __init__(self, obj):
        self.idx_obj = obj

    def _normalize_channel_key(self, key):

        if not isinstance(key, tuple):
            # If only one index is handed over, convert key to tuple
            key = (key,)

        channel_ndim = max(self.idx_obj.ndim - 1, 0)
        if channel_ndim == 0:
            return tuple()

        if key.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis")

        normalized = []
        for item in key:
            if item is Ellipsis:
                remaining = channel_ndim - (len(key) - 1)
                normalized.extend([slice(None)] * max(remaining, 0))
            else:
                normalized.append(item)

        if len(normalized) > channel_ndim:
            raise IndexError(
                f"too many channel indices for signal with {channel_ndim} "
                f"channel dimension{'s' if channel_ndim != 1 else ''}"
            )

        if len(normalized) < channel_ndim:
            normalized.extend([slice(None)] * (channel_ndim - len(normalized)))

        return tuple(normalized)

    def _channel_index(self, key):

        if np.ndim(self.idx_obj) == 1:
            return slice(None, None, None)
        return (slice(None, None, None),) + self._normalize_channel_key(key)

    def _normalize_channel_view(self, out):

        if not isinstance(out, np.ndarray):
            return out

        if out.ndim == 1:
            return out[:, np.newaxis]

        if out.ndim > 2 and np.prod(out.shape[1:]) == 1:
            return out.reshape(out.shape[0], 1)

        return out

    def __getitem__(self, key) -> Any:

        idx = self._channel_index(key)
        return self._normalize_channel_view(self.idx_obj[idx])

    def __setitem__(self, key, value):
        idx = self._channel_index(key)
        self.idx_obj[idx] = value
        return self.idx_obj
