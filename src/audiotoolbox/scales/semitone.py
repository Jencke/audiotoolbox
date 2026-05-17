import numpy as np

from .base import ScaleBase


class SemitoneScale(ScaleBase):
    """Semitone scale based on a configurable reference note and frequency."""

    def from_freq(self, frequency, ref_freq: float = 440.0, ref_note: float = 69.0):
        """Convert frequency in Hz to semitone index (MIDI-like note number)."""
        frequency = np.asarray(frequency, dtype=float)
        scalar_input = frequency.ndim == 0
        frequency = np.atleast_1d(frequency)
        if np.any(frequency <= 0):
            raise ValueError("frequency must be > 0 Hz")
        notes = ref_note + 12.0 * np.log2(frequency / ref_freq)
        return float(notes[0]) if scalar_input else notes

    def to_freq(self, scale_value, ref_freq: float = 440.0, ref_note: float = 69.0):
        """Convert semitone index (MIDI-like note number) to frequency in Hz."""
        scale_value = np.asarray(scale_value, dtype=float)
        scalar_input = scale_value.ndim == 0
        scale_value = np.atleast_1d(scale_value)
        freq = ref_freq * (2.0 ** ((scale_value - ref_note) / 12.0))
        return float(freq[0]) if scalar_input else freq

    def get_bw(self, fc):
        """Bandwidth in Hz for a 1-semitone interval centered at fc."""
        fc = np.asarray(fc, dtype=float)
        scalar_input = fc.ndim == 0
        fc = np.atleast_1d(fc)
        if np.any(fc <= 0):
            raise ValueError("fc must be > 0 Hz")
        ratio = 2.0 ** (1.0 / 24.0)
        bw = fc * (ratio - 1.0 / ratio)
        return float(bw[0]) if scalar_input else bw


scale = SemitoneScale()
