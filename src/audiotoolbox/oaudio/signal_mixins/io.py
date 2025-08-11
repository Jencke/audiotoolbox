"""Signal mixins for organizing Signal class functionality."""

from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from ... import audiotoolbox as audio, io

if TYPE_CHECKING:
    from ..signal import Signal


class IOMixin:
    """Mixin for signal io methods."""

    def from_file(self, filename: str, start: int = 0, channels="all"):
        """
        Load a signal from an audio file.

        This method loads a signal from an audio file and assigns it to the current
        Signal object. The signal can be loaded from a specific start point and for
        specific channels.

        Parameters
        ----------
        filename : str
            The path to the audio file to load.
        start : int, optional
            The starting sample index from which to load the signal. Default is 0.
        channels : int, tuple, or str, optional
            The channels to load from the audio file. Can be an integer specifying
            a single channel, a tuple specifying multiple channels, or "all" to load
            all channels. Default is "all".

        Returns
        -------
        Signal
            The Signal object with the loaded audio data.

        Raises
        ------
        ValueError
            If the number of channels in the loaded signal does not match the number
            of channels in the current Signal object.

        Examples
        --------
        Load a signal from a file starting at the beginning and using all channels:

        >>> sig = Signal(2, 1, 48000)
        >>> sig.from_file("example.wav")

        Load a signal from a file starting at sample index 1000 and using the first channel:

        >>> sig = Signal(1, 1, 48000)
        >>> sig.from_file("example.wav", start=1000, channels=0)
        """
        sig = audio.from_file(filename, start=start, stop=self.n_samples + start)
        if channels == "all":
            channels = slice(None)

        # Convert channels to a tuple if it's not already
        if not isinstance(channels, tuple):
            channels = (channels,)

        sig = sig.ch[channels]
        print(sig.shape)
        print(self.shape)
        if sig.n_channels != self.n_channels:
            raise ValueError("Number of channels must match.")

        self[:] = sig
        return self

    def write_file(self, filename, **kwargs):
        """
        Save the signal as an audio file.

        This method saves the current signal as an audio file. Additional parameters
        for the file format can be specified through keyword arguments. The file can
        be saved in any format supported by libsndfile, such as WAV, FLAC, AIFF, etc.

        Parameters
        ----------
        filename : str
            The filename to save the audio file as.
        **kwargs
            Additional keyword arguments to be passed to the `audiotoolbox.wav.writefile`
            function. These can include format and subtype.

        Returns
        -------
        None

        Examples
        --------
        Save the signal to a file named "output.wav":

        >>> sig = Signal(2, 1, 48000)
        >>> sig.write_file("output.wav")

        Save the signal to a file with a specific format and subtype:

        >>> sig = Signal(2, 1, 48000)
        >>> sig.write_file("output.wav", format="WAV", subtype="PCM_16")

        Save the signal to a FLAC file:

        >>> sig = Signal(2, 1, 48000)
        >>> sig.write_file("output.flac", format="FLAC")

        See Also
        --------
        audiotoolbox.wav.writefile : Function used to write the audio file.
        """
        io.write_file(filename, self, self.fs, **kwargs)

    def play(self, block: bool = True):
        """Quick playback of the signal over the default audio output device.

        Parameters
        ----------
        block : bool, optional
            If True, the method will block until playback is finished. If False,
            playback will be non-blocking and the method will return immediately.
            Default is True.
        """

        sd.play(self, samplerate=self.fs)
        if block:
            sd.wait()
