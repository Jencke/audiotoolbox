"""Signal mixins for organizing Signal class functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union
import numpy as np
import warnings
from .. import core as audio, filter as filt

if TYPE_CHECKING:
    from ..signal import Signal


class FilteringMixin:
    """Mixin for signal filtering methods."""

    def bandpass(self, fc, bw, filter_type, **kwargs):
        r"""Apply a bandpass filter.

        Applies a bandpass filter to the signal. The available filters
        are:

        - brickwall: A 'optimal' brickwall filter
        - gammatone: A real valued gammatone filter
        - butter: A butterworth filter

        For additional filter parameters and detailed description see
        the respective implementations:

        - :meth:`audiotoolbox.filter.brickwall`
        - :meth:`audiotoolbox.filter.gammatone`
        - :meth:`audiotoolbox.filter.butterworth`

        Parameters
        ----------
        fc : scalar
            The bandpass center frequency in Hz
        bw : scalar
            The filter bandwidth in Hz
        filter_type : {'brickwall', 'gammatone', 'butter'}
            The filtertype
        **kwargs :
            Further keyword arguments are passed to the respective
            filter functions

        Returns
        --------
            Returns itself : Signal

            When a complex-valued output is requested (e.g. gammatone
            with ``return_complex=True``), a new complex Signal is
            returned and a UserWarning is emitted. In-place conversion
            from real to complex is not possible without reallocating
            the underlying ndarray buffer.

            If you want explicit control, cast first and call bandpass
            on the complex signal:

            ``complex_signal = signal.astype(complex)``
            ``complex_signal = complex_signal.bandpass(..., return_complex=True)``

        See Also
        --------
        audiotoolbox.filter.brickwall
        audiotoolbox.filter.gammatone
        audiotoolbox.filter.butterworth
        """
        # Default gammatone to real valued implementation
        if filter_type == "gammatone":
            if "return_complex" not in kwargs:
                kwargs["return_complex"] = False

        filt_signal = filt.bandpass(self, fc, bw, filter_type, **kwargs)

        # Complex output cannot be represented in-place on the existing
        # real-valued ndarray without corrupting its buffer layout.
        if np.iscomplexobj(filt_signal):
            warnings.warn(
                "bandpass with complex output returns a new Signal instead of modifying in-place",
                UserWarning,
                stacklevel=2,
            )
            complex_signal = self.astype(complex)
            complex_signal[:] = filt_signal
            return complex_signal

        self[:] = filt_signal

        return self

    def lowpass(self, f_cut, filter_type, **kwargs):
        """Apply a lowpass filter to the Signal.

        This function provides a unified interface to all lowpass
        filters implemented in audiotoolbox.

        - brickwall: A 'optimal' brickwall filter
        - butter: A butterworth filter

        For additional filter parameters and detailed description see
        the respective implementations:

        - :meth:`audiotoolbox.filter.brickwall`
        - :meth:`audiotoolbox.filter.butterworth`

        Parameters
        ----------
        signal : ndarray or Signal
            The input signal.
        f_cut : float
            The cutoff frequency in Hz
        filter_type : {'butter', 'brickwall'}
            The filter type
        fs : None or int
            The sampling frequency, must be provided if not using the Signal class.
        **kwargs :
            Further arguments such as 'order' that are passed to the filter functions.

        Returns
        -------
        Signal : The filtered Signal

        See Also
        --------
        audiotoolbox.filter.brickwall
        audiotoolbox.filter.butterworth

        """
        filt_signal = filt.lowpass(self, f_cut, filter_type, **kwargs)

        self[:] = filt_signal
        return self

    def highpass(self, f_cut, filter_type, **kwargs):
        """Apply a highpass filter to the Signal.

        This function provides a unified interface to all highpass
        filters implemented in audiotoolbox.

        - brickwall: A 'optimal' brickwall filter
        - butter: A butterworth filter

        For additional filter parameters and detailed description see
        the respective implementations:

        - :meth:`audiotoolbox.filter.brickwall`
        - :meth:`audiotoolbox.filter.butterworth`

        Parameters
        ----------
        signal : ndarray or Signal
            The input signal.
        f_cut : float
            The cutoff frequency in Hz
        filter_type : {'butter', 'brickwall'}
            The filter type
        fs : None or int
            The sampling frequency, must be provided if not using the
            Signal class.
        **kwargs :
            Further arguments such as 'order' that are passed to the
            filter functions.

        Returns
        -------
        Signal : The filtered Signal

        See Also
        --------
        audiotoolbox.filter.brickwall
        audiotoolbox.filter.butterworth

        """
        filt_signal = filt.highpass(self, f_cut, filter_type, **kwargs)

        self[:] = filt_signal
        return self

    def convolve(
        self: "Signal",
        kernel: Union["Signal", np.ndarray],
        mode: Literal["full", "valid", "same"] = "full",
        overlap_dimensions: bool = True,
    ) -> "Signal":
        r"""Convolve the signal with the kernel **in place** and return self.

        .. warning::
            This method modifies the signal in place, including resizing it.
            Any other variable that holds a reference to this signal will see
            the mutated data after this call.  To get a new Signal without
            touching the original, use :meth:`convolved` instead::

                result = signal.convolved(kernel)

            Calling this method on a view or slice (i.e. any Signal that does
            not own its data) raises a ``RuntimeError``.  Call ``.copy()``
            first if you need to convolve a slice in place.

        The convolution is performed along the overlapping dimensions of the
        two signals. If the signal has two channels and the kernel has two
        channels, each channel of the signal is convolved with the matching
        channel of the kernel, and the result again has two channels.
        If `overlap_dimensions` is False, every signal channel is convolved
        with every kernel channel, producing an output whose channel shape is
        the outer product of the two channel shapes.

        Parameters
        ----------
        kernel : Signal or ndarray
            The kernel to convolve with.
        mode : {'full', 'valid', 'same'}, optional
            The convolution mode (default = ``'full'``).
        overlap_dimensions : bool, optional
            Whether to convolve only along overlapping dimensions (default =
            ``True``).

        Returns
        -------
        Signal
            ``self`` after in-place modification — suitable for method
            chaining.

            Exception: if the kernel is complex while the signal is real, the
            result cannot be stored in the real-valued buffer in place. A new
            complex ``Signal`` is returned instead and a ``UserWarning`` is
            emitted (the same constraint as :meth:`bandpass`).

        Raises
        ------
        RuntimeError
            If ``self`` does not own its data (e.g. it is a slice or the
            result of a ufunc).  Call ``.copy()`` first.

        Examples
        --------
        If the last dimension of signal and the first dimension of kernel match,
        convolution takes place along this axis. This means that the first
        channel of the signal is convolved with the first channel of the kernel,
        the second with the second.

        >>> signal = Signal(2, 1, 48000)
        >>> kernel = Signal(2, 100e-3, 48000)
        >>> signal.convolve(kernel)
        >>> signal.n_channels
        2

        This also works with multiple overlapping dimensions.

        >>> signal = Signal((5, 2, 3), 1, 48000)
        >>> kernel = Signal((2, 3), 100e-3, 48000)
        >>> signal.convolve(kernel)
        >>> signal.n_channels
        (5, 2, 3)

        The 'overlap_dimensions' keyword can be set to False if all signal
        channels are instead convolved with all kernels.

        >>> signal = Signal(2, 1, 48000)
        >>> kernel = Signal(2, 100e-3, 48000)
        >>> signal.convolve(kernel, overlap_dimensions=False)
        >>> signal.n_channels
        (2, 2)

        """
        if not self.flags.owndata:
            raise RuntimeError(
                "convolve resizes the signal in-place and cannot be called on "
                "a view or slice. Call .copy() first, or use .convolved() for "
                "a non-mutating alternative."
            )
        # Accept a plain ndarray kernel as documented; wrap it at the
        # signal's sampling rate. A Signal is returned unchanged.
        kernel = audio.as_signal(kernel, self.fs)
        fs = self.fs
        dim_sig = self.channel_shape
        dim_kernel = kernel.channel_shape

        # Squeeze the last dimension if it is 1
        squeeze_idx_k = ()
        squeeze_idx_sig = ()
        if dim_kernel[-1] == 1:
            dim_kernel = dim_kernel[:-1]
            squeeze_idx_k = (0,)
        if dim_sig[-1] == 1:
            dim_sig = dim_sig[:-1]
            squeeze_idx_sig = (0,)

        # Determine if some of the dimensions overlap. This is computed on the
        # *squeezed* shapes: counting a trailing singleton axis (that is then
        # squeezed away) as overlapping would leave the reshape/broadcast below
        # inconsistent.
        if overlap_dimensions:
            dim_overlap = audio._get_dim_overlap(dim_sig, dim_kernel)
        else:
            dim_overlap = 0

        new_nch = (*dim_sig, *dim_kernel[dim_overlap:])
        if mode == "same":
            new_nsamp = self.n_samples
        elif mode == "full":
            new_nsamp = self.n_samples + kernel.n_samples - 1
        elif mode == "valid":
            new_nsamp = self.n_samples - kernel.n_samples + 1
        else:
            raise ValueError("mode not implemented")
        # Promote the output dtype so a complex kernel (or signal) keeps its
        # imaginary part instead of being silently truncated.
        out_dtype = np.result_type(self.dtype, kernel.dtype)
        new_signal = audio.Signal(new_nch, new_nsamp / fs, fs, dtype=out_dtype)

        # Vectorized FFT convolution – replaces the O(n_sig * n_kernel) Python loop.
        #
        # Strategy: reshape sig and kernel so that their outer (non-overlapping)
        # dims broadcast against each other, then run a single batched FFT.
        #
        #   sig:  (T, *sig_outer, *overlap) → (T, *sig_outer, *overlap, 1…)
        #   ker:  (T, *overlap, *k_outer)   → (T, 1…, *overlap, *k_outer)
        #   product:                           (T, *sig_outer, *overlap, *k_outer)
        #
        # The overlap dims multiply element-wise (correct for per-channel
        # convolution); the outer dims multiply via singleton broadcasting
        # (correct for the cross-product case).

        sig_arr = np.asarray(self)
        ker_arr = np.asarray(kernel)

        # Apply the same trailing-singleton squeeze used by the original code,
        # but only when the array is 2-D or higher (has an explicit channel
        # axis).  For 1-D signals the sample axis IS the only axis; indexing
        # [..., 0] would return a scalar rather than the full time series.
        if squeeze_idx_sig and sig_arr.ndim > 1:
            sig_arr = sig_arr[..., 0]
        if squeeze_idx_k and ker_arr.ndim > 1:
            ker_arr = ker_arr[..., 0]

        # Number of outer (non-overlapping) dims on each side.
        # Use max(0, ...) because after squeezing, len(dim_X) can be < dim_overlap.
        n_sig_outer = max(0, len(dim_sig) - dim_overlap) if dim_overlap > 0 else len(dim_sig)
        n_k_outer = max(0, len(dim_kernel) - dim_overlap)

        sig_arr = sig_arr.reshape(sig_arr.shape + (1,) * n_k_outer)
        ker_arr = ker_arr.reshape(
            ker_arr.shape[:1] + (1,) * n_sig_outer + ker_arr.shape[1:]
        )

        n_fft = int(2 ** np.ceil(np.log2(self.n_samples + kernel.n_samples - 1)))

        # convolve in frequency domain, using real FFT if both inputs are real-valued
        # use complex FFT if either input is complex-valued
        if np.isrealobj(sig_arr) and np.isrealobj(ker_arr):
            raw = np.fft.irfft(
                np.fft.rfft(sig_arr, n=n_fft, axis=0)
                * np.fft.rfft(ker_arr, n=n_fft, axis=0),
                n=n_fft,
                axis=0,
            )
        else:
            raw = np.fft.ifft(
                np.fft.fft(sig_arr, n=n_fft, axis=0)
                * np.fft.fft(ker_arr, n=n_fft, axis=0),
                axis=0,
            )

        # Trim to the requested output length.
        if mode == "full":
            new_signal[:] = raw[:new_nsamp]
        elif mode == "same":
            start = (kernel.n_samples - 1) // 2
            new_signal[:] = raw[start : start + new_nsamp]
        elif mode == "valid":
            start = kernel.n_samples - 1
            new_signal[:] = raw[start : start + new_nsamp]

        # A complex result cannot be represented in a real-valued signal in
        # place without reallocating the underlying buffer (same constraint as
        # Signal.bandpass). Return a new complex Signal instead of truncating.
        if np.iscomplexobj(new_signal) and not np.iscomplexobj(self):
            warnings.warn(
                "convolve produced a complex result and returns a new Signal "
                "instead of modifying in place",
                UserWarning,
                stacklevel=2,
            )
            return new_signal

        self.resize(new_signal.shape, refcheck=False)
        self[:] = new_signal
        return self

    def convolved(
        self: "Signal",
        kernel: Union["Signal", np.ndarray],
        mode: Literal["full", "valid", "same"] = "full",
        overlap_dimensions: bool = True,
    ) -> "Signal":
        """Return a new Signal convolved with the kernel, leaving self unchanged.

        This is the non-mutating counterpart of :meth:`convolve`.  It is
        equivalent to ``self.copy().convolve(kernel, mode, overlap_dimensions)``
        but makes the intent explicit.

        Parameters
        ----------
        kernel : Signal or ndarray
            The kernel to convolve with.
        mode : {'full', 'valid', 'same'}, optional
            The convolution mode (default = ``'full'``).
        overlap_dimensions : bool, optional
            Whether to convolve only along overlapping dimensions (default =
            ``True``).

        Returns
        -------
        Signal
            A new Signal containing the convolution result.
        """
        return self.copy().convolve(kernel, mode=mode, overlap_dimensions=overlap_dimensions)
