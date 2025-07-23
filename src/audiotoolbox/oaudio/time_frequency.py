from typing import TYPE_CHECKING

from .. import audiotoolbox as audio
import numpy as np
from scipy.signal.windows import get_window
from scipy.signal import spectrogram

if TYPE_CHECKING:
    from ...audiotoolbox.filter.bank.filterbank import FilterBank
    from ...audiotoolbox.oaudio.signal import Signal


class TimeFrequency(object):
    """Class containing time-frequency analysis methods."""

    def __init__(self, sig):
        self.sig = sig

    def octave_band_specgram(
        self, nperseg: int = 1024, noverlap: int = 512, win: str = "hann", **kwargs
    ) -> "tuple[Signal, np.ndarray]":
        """Calculate the octave band spectrogram of a signal.

        This function applies an octave filter bank to the signal and computes the spectrogram.

        Parameters
        ----------
        nperseg : int, optional
            The number of samples per segment for the spectrogram (default is 1024).
        noverlap : int, optional
            The number of samples to overlap between segments (default is 512).
        win : str, optional
            The window function to apply (default is 'hann'). Can be any valid window function name recognized by `scipy.signal.get_window`.
        **kwargs : dict, optional
            Additional parameters to pass to the octave bank function.

        Returns
        -------
        tuple[audio.Signal, np.ndarray]
            A tuple containing the spectrogram as an audio Signal in dBFS and the center frequencies of the octave bands

        """

        bank = audio.filter.bank.octave_bank(fs=self.sig.fs, **kwargs)
        filt_sig = bank.filt(self.sig)
        blocked_sig = filt_sig.as_blocked(block_size=nperseg, overlap=noverlap)
        win_func = get_window(win, blocked_sig.n_samples, fftbins=True)

        # Calculate the sum of squares of the window values
        g_energy = np.sum(win_func**2) / len(win_func)
        db_correction = -10 * np.log10(g_energy)
        blocked_sig = blocked_sig * win_func[:, np.newaxis, np.newaxis]
        spec = blocked_sig.stats.dbfs
        spec._fs = (blocked_sig.shape[1] - 1) / self.sig.duration
        spec = spec + db_correction
        fc = bank.fc
        return spec, fc

    def gammatone_specgram(
        self, nperseg: int = 1024, noverlap: int = 512, win: str = "hann", **kwargs
    ) -> "tuple[Signal, np.ndarray]":
        """Calculate the gammatone spectrogram of a signal.

        This function applies a gammatone filter bank to the signal and computes the spectrogram.

        Parameters
        ----------
        nperseg : int, optional
            The number of samples per segment for the spectrogram (default is 1024).
        noverlap : int, optional
            The number of samples to overlap between segments (default is 512).
        win : str, optional
            The window function to apply (default is 'hann'). Can be any valid window function name recognized by `scipy.signal.get_window`.
        **kwargs : dict, optional
            Additional parameters to pass to the auditory gamma bank function.

        Returns
        -------
        tuple[audio.Signal, np.ndarray]
            A tuple containing the spectrogram as an audio Signal in dBFS and the center frequencies of the gammatone filters.

        """

        bank = audio.filter.bank.auditory_gamma_bank(fs=self.sig.fs, **kwargs)
        # .real is valid since Signal inherits from numpy.ndarray
        filt_sig = bank.filt(self.sig).real.copy()  # type: ignore
        blocked_sig = filt_sig.as_blocked(block_size=nperseg, overlap=noverlap)
        win_func = get_window(win, blocked_sig.n_samples, fftbins=True)

        # Calculate the sum of squares of the window values
        g_energy = np.sum(win_func**2) / len(win_func)
        db_correction = -10 * np.log10(g_energy)
        blocked_sig = blocked_sig * win_func[:, np.newaxis, np.newaxis]
        spec = blocked_sig.stats.dbfs
        spec._fs = (blocked_sig.shape[1] - 1) / self.sig.duration
        spec = spec + db_correction
        fc = bank.fc
        return spec, fc

    def filterbank_specgram(
        self,
        bank: "FilterBank",
        nperseg: int = 1024,
        noverlap: int = 512,
        win: str = "hann",
    ) -> "tuple[Signal, np.ndarray]":
        """Calculate the spectrogram of a signal using a specified filter bank.

        This function applies a filter bank to the signal and computes the spectrogram.

        Parameters
        ----------
        bank : FilterBank
            The filter bank to apply to the signal.
        nperseg : int, optional
            The number of samples per segment for the spectrogram (default is 1024).
        noverlap : int, optional
            The number of samples to overlap between segments (default is 512).
        win : str, optional
            The window function to apply (default is 'hann'). Can be any valid window function name recognized by `scipy.signal.get_window`.

        Returns
        -------
        tuple[audio.Signal, np.ndarray]
            A tuple containing the spectrogram as an audio Signal in dBFS and the center frequencies of the filter bank.
        """

        filt_sig = bank.filt(self.sig).real.copy()  # type: ignore
        blocked_sig = filt_sig.as_blocked(block_size=nperseg, overlap=noverlap)
        win_func = get_window(win, blocked_sig.n_samples, fftbins=True)

        # Calculate the sum of squares of the window values
        g_energy = np.sum(win_func**2) / len(win_func)
        db_correction = -10 * np.log10(g_energy)
        blocked_sig = blocked_sig * win_func[:, np.newaxis, np.newaxis]
        spec = blocked_sig.stats.dbfs
        spec._fs = (blocked_sig.shape[1] - 1) / self.sig.duration
        spec = spec + db_correction
        fc = bank.fc
        return spec, fc

    def stft_specgram(
        self, nperseg: int = 1024, noverlap: int = 512, win: str = "hann", **kwargs
    ) -> "tuple[Signal, np.ndarray]":
        """Calculate the Short-Time Fourier Transform (STFT) spectrogram of a signal.

        This function computes the STFT of the signal and returns the spectrogram. It is a wrapper around the `scipy.signal.spectrogram` function.

        Parameters
        ----------
        nperseg : int, optional
            The number of samples per segment for the STFT (default is 1024).
        noverlap : int, optional
            The number of samples to overlap between segments (default is 512).
        win : str, optional
            The window function to apply (default is 'hann'). Can be any valid window function name recognized by `scipy.signal.get_window`.
        **kwargs : dict, optional
            Additional parameters to pass to the `spectrogram` function.

        Returns
        -------
        tuple[audio.Signal, np.ndarray]
            A tuple containing the spectrogram as an audio Signal in dBFS and the center frequencies of the STFT.

        """
        f, t, spec = spectrogram(
            self.sig.data,
            fs=self.sig.fs,
            window=win,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="spectrum",
            **kwargs,
        )
        spec = audio.as_signal(spec.T, fs=self.sig.fs)
        spec._fs = (spec.shape[1] - 1) / self.sig.duration
        fc = f
        spec = 10 * np.log10(np.abs(spec) / 0.5)
        return spec, fc
