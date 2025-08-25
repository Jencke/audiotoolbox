import numpy as np

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
        import matplotlib.pyplot as plt

        assert self.sig.n_channels == 1, "Only single channel signals supported"

        # make sure that oct_fraction is the same for spectrogram and band_levels
        if "oct_fraction" in specgram_args:
            oct_fraction = specgram_args["oct_fraction"]
        else:
            oct_fraction = 3

        spec, freq = self.sig.time_frequency.octave_band_specgram(**specgram_args)
        bandlevels, freq = self.sig.stats.octave_band_levels(oct_fraction=oct_fraction)

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
        ax[1, 0].pcolormesh(spec.time, freq, spec.T)
        ax[0, 1].set_visible(False)
        ax[1, 0].set_yscale("log")
        ax[1, 0].set_xlabel("Time / s")
        ax[1, 0].set_yticks(freq[::3])
        ax[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(freq_formatter))
        ax[1, 0].set_ylabel("Frequency / Hz")
        ax[1, 1].barh(freq, bandlevels - basevalue, left=basevalue, height=0.15 * freq)
        ax[1, 1].set_xlabel("Level / dB FS")
        ax[1, 1].minorticks_off()
        dbfs = self.sig.stats.dbfs
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
        import matplotlib.pyplot as plt

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
