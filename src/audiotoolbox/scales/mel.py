import numpy as np

from .base import ScaleBase


class MelScale(ScaleBase):
    """Mel scale conversions using the HTK-style definition."""

    def from_freq(self, frequency):
        """Convert frequency in Hz to Mel."""
        frequency = np.asarray(frequency, dtype=float)
        scalar_input = frequency.ndim == 0
        frequency = np.atleast_1d(frequency)
        if np.any(frequency < 0):
            raise ValueError("frequency must be >= 0 Hz")
        mel = 2595.0 * np.log10(1.0 + frequency / 700.0)
        return float(mel[0]) if scalar_input else mel

    def to_freq(self, scale_value):
        """Convert Mel values to frequency in Hz."""
        scale_value = np.asarray(scale_value, dtype=float)
        scalar_input = scale_value.ndim == 0
        scale_value = np.atleast_1d(scale_value)
        freq = 700.0 * (10.0 ** (scale_value / 2595.0) - 1.0)
        return float(freq[0]) if scalar_input else freq

    def get_bw(self, fc):
        """Bandwidth in Hz corresponding to a 1-Mel interval around fc."""
        fc = np.asarray(fc, dtype=float)
        scalar_input = fc.ndim == 0
        fc = np.atleast_1d(fc)
        if np.any(fc < 0):
            raise ValueError("fc must be >= 0 Hz")
        mel_center = self.from_freq(fc)
        upper = self.to_freq(mel_center + 0.5)
        lower = self.to_freq(np.maximum(mel_center - 0.5, 0.0))
        bw = upper - lower
        return float(bw[0]) if scalar_input else bw


scale = MelScale()
