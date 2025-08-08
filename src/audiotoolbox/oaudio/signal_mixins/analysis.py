"""Signal mixins for organizing Signal class functionality."""

from typing import TYPE_CHECKING
import numpy as np
import warnings
from ... import audiotoolbox as audio

if TYPE_CHECKING:
    from ..signal import Signal


class AnalysisMixin:
    """Mixin for signal analysis methods."""

    def as_blocked(self, block_size: int = 1024, overlap: int = 512):
        """Creates a blocked view of the signal.

        This method creates a blocked view of the signal, where each block
        has a fixed size and overlaps with the previous block by a specified amount.

        Parameters
        ----------
        block_size : int
            The size of each block in samples.
        overlap : int
            The amount of overlap between consecutive blocks in samples.

        Returns
        -------
        Signal
            A blocked view of the signal.
        """

        step = block_size - overlap
        current_length = self.n_samples

        required_length = (
            np.ceil((current_length - block_size) / step) * step + block_size
        ).astype(int)

        # If the original data is already long enough, no padding needed
        if required_length < current_length:
            required_length = current_length
        n_pad = required_length - current_length

        if n_pad > 0:
            warnings.warn(
                f"Zero padding {n_pad} samples to the end of the signal to create blocks.",
                UserWarning,
            )
            self.zeropad(number=[0, n_pad])

        padded_length = self.n_samples

        # Calculate the number of windows that can be formed
        # The formula is (total_length - window_size) // step + 1
        num_windows = (padded_length - block_size) // step + 1

        output_shape = (block_size, num_windows) + self.shape[1:]

        original_strides = self.strides
        itemsize = self.itemsize

        # Calculate new strides
        if self.ndim == 1:  # Mono signal (N_samples,)
            # Stride to next sample within a block: itemsize
            # Stride to next window: step * itemsize
            strides = (itemsize, step * itemsize)
        elif (
            self.ndim >= 2
        ):  # Multi-channel or higher dimensions (N_samples, N_channels, ...)
            # Stride to next sample within a block (across channels): original_strides[0]
            # Stride to next window: step * original_strides[0]
            # Strides for remaining dimensions (e.g., channels): original_strides[1:]
            strides = (
                original_strides[0],
                step * original_strides[0],
            ) + original_strides[1:]

        # strides = (step * itemsize, itemsize)

        overlapping_windows = np.lib.stride_tricks.as_strided(
            self, shape=output_shape, strides=strides
        )

        # overlapping_windows = as_strided(sig, shape=(num_windows, block_size), strides=strides)

        blocks = overlapping_windows.view(audio.Signal)
        blocks.__array_finalize__(self)

        return blocks
