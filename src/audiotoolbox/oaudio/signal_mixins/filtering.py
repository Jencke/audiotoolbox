"""Signal mixins for organizing Signal class functionality."""

from typing import TYPE_CHECKING, Literal
import numpy as np
from scipy.signal import fftconvolve
import warnings
from ... import audiotoolbox as audio, filter as filt, io

if TYPE_CHECKING:
    from ..signal import Signal


class FilteringMixin:
    """Mixin for signal filtering methods."""

    def bandpass(self, fc, bw, filter_type, **kwargs):
        r"""Apply a bandpass filter.

        Applies a bandpass filter to the signal. The availible filters
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
            The banddpass center frequency in Hz
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

        # in case of complex output, signal needs to be reshaped and
        # typecast
        if np.iscomplexobj(filt_signal):
            shape = self.shape
            self.dtype = complex
            self.resize(shape, refcheck=False)
        self[:] = filt_signal

        return self

    def lowpass(self, f_cut, filter_type, **kwargs):
        """Apply a lowpass filter to the Signal.

        This function provieds a unified interface to all lowpass
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

        This function provieds a unified interface to all highpass
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
        self,
        kernel,
        mode: Literal[
            "full",
            "valid",
            "same",
        ] = "full",
        overlap_dimensions: bool = True,
    ):
        r"""Convolves the current signal with the given kernel.

        This method performs a convolution operation between the current signal
        and the provided kernel. The convolution is performed along the
        overlapping dimensions of the two signals. E.g., If the signal has two channels
        and the kernel has two channels, the first channel of the signal is convolved
        with the first channel of the kernel, and the second channel of the signal is
        convolved with the second channel of the kernel. The resulting signal will again have
        two channels. If `overlap_dimensions` is False, the convolution is performed
        along all dimensions. A Signal with two channels convolved with a two-channel kernel
        will result in an output of shape (2, 2) where each channel of the signal is convolved with
        each channel of the kernel.

        this method uses scipy.Signal.fftconvolve for the convolution.

        Parameters
        ----------
        kernel : Signal
            The kernel to convolve with.
        mode : str {'full', 'valid', 'same'}, optional
            The convolution mode for fftconvolve (default=full)
        overlap_dimensions : bool, optional
            Whether to convolve only along overlapping dimensions. If True, the
            convolution is performed only along the dimensions that overlap between
            the two signals. If False, the convolution is performed along all
            dimensions. Defaults to True.

        Returns
        -------
        Self
            The convolved signal.

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
        fs = self.fs
        dim_sig = np.atleast_1d(self.n_channels)
        dim_kernel = np.atleast_1d(kernel.n_channels)

        # Determine if some of the dimension overlap
        if overlap_dimensions:
            dim_overlap = audio._get_dim_overlap(dim_sig, dim_kernel)
        else:
            dim_overlap = 0

        # Squeeze the last dimension if it is 1
        squeeze_idx_k = ()
        squeeze_idx_sig = ()
        if dim_kernel[-1] == 1:
            dim_kernel = dim_kernel[:-1]
            squeeze_idx_k = (0,)
        if dim_sig[-1] == 1:
            dim_sig = dim_sig[:-1]
            squeeze_idx_sig = (0,)

        new_nch = (*dim_sig, *dim_kernel[dim_overlap:])
        if mode == "same":
            new_nsamp = self.n_samples
        elif mode == "full":
            new_nsamp = self.n_samples + kernel.n_samples - 1
        elif mode == "valid":
            new_nsamp = self.n_samples - kernel.n_samples + 1
        else:
            raise ValueError("mode not implemented")
        new_signal = audio.Signal(new_nch, new_nsamp / fs, fs)

        if dim_overlap != 0:
            n_sig = np.prod(dim_sig[:-dim_overlap])
        else:
            n_sig = np.prod(dim_sig)
        n_kernel = np.prod(dim_kernel[dim_overlap:])
        for i_sig in range(n_sig):
            for i_k in range(n_kernel):
                # only indices that do not overlap need to be looked at
                if dim_overlap != 0:
                    idx_sig = np.unravel_index(i_sig, dim_sig[:-dim_overlap])
                else:
                    idx_sig = np.unravel_index(i_sig, dim_sig)
                idx_k = np.unravel_index(i_k, dim_kernel[dim_overlap:])

                overlap_slice = (slice(None, None, None),) * dim_overlap
                idx_sig_combined = idx_sig + overlap_slice + squeeze_idx_sig
                idx_k_combined = overlap_slice + idx_k + squeeze_idx_k

                a = self.ch[idx_sig_combined]
                b = kernel.ch[idx_k_combined]
                newsig_idx = idx_sig + overlap_slice + idx_k

                new_signal.ch[newsig_idx] = fftconvolve(a, b, mode=mode, axes=0)
        self.resize(new_signal.shape, refcheck=False)
        self[:] = new_signal
        return self
