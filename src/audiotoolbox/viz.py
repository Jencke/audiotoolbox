import numpy as np
import matplotlib.pyplot as plt
from typing import Union

COLOR_R = "#d65c5c"
COLOR_L = "#5c5cd6"


# if TYPE_CHECKING:
#     from ...audiotoolbox.filter.bank.filterbank import FilterBank
#     from ...audiotoolbox.oaudio.signal import Signal


class Visualization(object):
    """Class containing time-frequency analysis methods."""

    def __init__(self, sig):
        self.sig = sig

    def specgram_overview(self, specgram_args: dict = {}) -> tuple:
        """Plot a signal overview with octave band spectrogram, signal and octave band levels.

        This function creates a figure with three subplots: the time-domain signal,
        the octave band spectrogram, and the octave band levels.

        Parameters
        ----------
        specgram_args : dict
            Arguments to be passed to the spectrogram function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plots.
        ax : numpy.ndarray
            The array of axes objects for the subplots.
        """

        assert self.sig.n_channels == 1, "Only single channel signals supported"

        # make sure that oct_fraction is the same for spectrogram and band_levels
        if "oct_fraction" in specgram_args:
            oct_fraction = specgram_args["oct_fraction"]
        else:
            oct_fraction = 3

        spec, spec_freq = self.sig.time_frequency.octave_band_specgram(**specgram_args)
        freq, bandlevels = self.sig.stats.octave_band_levels(oct_fraction=oct_fraction)

        basevalue = bandlevels.min() * 1.1

        def freq_formatter(x, pos):
            return f"{x:.0f}"

        fig, ax = plt.subplots(
            2,
            2,
            height_ratios=[0.5, 1],
            width_ratios=[1, 0.5],
            sharex="col",
            sharey="row",
        )
        ax[0, 0].plot(self.sig.time, self.sig)
        ax[0, 0].set_ylabel("Amplitude")
        ax[1, 0].pcolormesh(spec.time, spec_freq, spec.T)
        ax[0, 1].set_visible(False)
        ax[1, 0].set_yscale("log")
        ax[1, 0].set_xlabel("Time / s")
        ax[1, 0].set_yticks(freq[::3])
        ax[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(freq_formatter))
        ax[1, 0].set_ylabel("Frequency / Hz")
        ax[1, 1].barh(freq, bandlevels - basevalue, left=basevalue, height=0.15 * freq)
        ax[1, 1].set_xlabel("Level / dB FS")
        ax[1, 1].minorticks_off()
        dbfs = self.sig.stats.dbfs[0]
        duration = self.sig.duration
        samples = self.sig.n_samples
        max_val = self.sig.max()
        fig.text(
            0.7,
            0.8,
            f"Duration:{duration:.1f} s\nSamples: {samples}\nLevel: {dbfs:.1f} dB FS\nMax: {max_val:.1f}",
        )
        fig.tight_layout()

        return fig, ax

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

        assert np.ndim(self.sig) <= 2, "Only 1 dimensional channel shapes allowed"

        if not ax:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.figure
        if self.sig.n_channels == 2:
            ax.plot(self.sig.time, self.sig[:, 0], color=COLOR_L)
            ax.plot(self.sig.time, self.sig[:, 1], color=COLOR_R)
        else:
            ax.plot(self.sig.time, self.sig)
        ax.set_xlabel("Time / s")
        ax.set_ylabel("Amplitude")
        return fig, ax

    def spectrum(
        self,
        single_sided: bool = True,
        minx: float = 20.0,
        maxx: float = 20000.0,
        power: bool = False,
        in_db: bool = True,
        ax: Union[None, plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the spectrum of the Signal using matplotlib.

        This function computes and plots the amplitude spectrum of the signal.

        Parameters
        ----------
        single_sided : bool
            If True, only the positive frequencies are plotted.
            (default is True)
        minx : float
            Minimum x-axis value in Hz. (default is 20.0).
        maxx : float
            Maximum x-axis value in Hz. (default is 20000.0). Values above
            the Nyquist frequency are clamped to the Nyquist frequency.
        power : bool
            If True, the power spectrum is plotted instead of the amplitude
            spectrum. (default is False)
        in_db : bool
            If True, the amplitude/power values are converted to dB scale.
            (default is True)
        ax : None, matplotlib.axis (optional)
            The axis that should be used for plotting. If None, a new
            figure is created. (default is None)

        """
        import matplotlib.pyplot as plt

        nyquist = self.sig.fs / 2.0
        if maxx > nyquist:
            maxx = nyquist

        if not ax:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.figure

        fsig = self.sig.to_freqdomain()
        freq = fsig.freq
        amplitude = np.abs(fsig)
        if power:
            amplitude = amplitude**2

        if single_sided:
            half_n = len(freq) // 2
            freq = freq[:half_n]
            amplitude = amplitude[:half_n]
            amplitude *= 2  # compensate for single sided spectrum

        if in_db:
            if power:
                amplitude = 10 * np.log10(amplitude + 1e-12)
            else:
                amplitude = 20 * np.log10(amplitude + 1e-12)

        ax.plot(freq, amplitude)
        ax.set_xlabel("Frequency / Hz")
        if not in_db:
            ax.set_ylabel("Power" if power else "Amplitude")
        else:
            ax.set_ylabel("Power / dB" if power else "Amplitude / dB")
        ax.set_xscale("log")
        ax.set_xlim(minx, maxx)
        return fig, ax
