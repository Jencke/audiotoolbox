import numpy as np

from .base import ScaleBase


class SemitoneScale(ScaleBase):
    """Semitone scale based on a configurable reference note and frequency."""

    def from_freq(self, frequency, ref_freq: float = 440.0, ref_note: float = 69.0):
        r"""Frequency to semitone index conversion.

        Converts frequency in Hz to a continuous MIDI-like semitone index:

        .. math:: n = n_{ref} + 12\log_2(f/f_{ref})

        Parameters
        ----------
        frequency : scalar or ndarray
            Frequency in Hz. Values must be strictly positive.
        ref_freq : float, optional
            Reference frequency in Hz. Default is ``440.0`` (A4).
        ref_note : float, optional
            Reference note number for ``ref_freq``. Default is ``69.0``.

        Returns
        -------
        scalar or ndarray
            Continuous semitone indices.
        """
        frequency = np.asarray(frequency, dtype=float)
        scalar_input = frequency.ndim == 0
        frequency = np.atleast_1d(frequency)
        if np.any(frequency <= 0):
            raise ValueError("frequency must be > 0 Hz")
        notes = ref_note + 12.0 * np.log2(frequency / ref_freq)
        return float(notes[0]) if scalar_input else notes

    def to_freq(self, scale_value, ref_freq: float = 440.0, ref_note: float = 69.0):
        r"""Semitone index to frequency conversion.

        Converts continuous semitone indices to frequency in Hz:

        .. math:: f = f_{ref} 2^{(n - n_{ref})/12}

        Parameters
        ----------
        scale_value : scalar or ndarray
            Continuous semitone indices.
        ref_freq : float, optional
            Reference frequency in Hz. Default is ``440.0`` (A4).
        ref_note : float, optional
            Reference note number for ``ref_freq``. Default is ``69.0``.

        Returns
        -------
        scalar or ndarray
            Frequencies in Hz.
        """
        scale_value = np.asarray(scale_value, dtype=float)
        scalar_input = scale_value.ndim == 0
        scale_value = np.atleast_1d(scale_value)
        freq = ref_freq * (2.0 ** ((scale_value - ref_note) / 12.0))
        return float(freq[0]) if scalar_input else freq

    def get_bw(self, fc):
        r"""Bandwidth for a 1-semitone interval.

        Calculates the frequency bandwidth in Hz corresponding to a
        1-semitone interval centered at ``fc``.

        Parameters
        ----------
        fc : scalar or ndarray
            Center frequency in Hz. Values must be strictly positive.

        Returns
        -------
        scalar or ndarray
            Bandwidth in Hz for a 1-semitone interval around ``fc``.
        """
        fc = np.asarray(fc, dtype=float)
        scalar_input = fc.ndim == 0
        fc = np.atleast_1d(fc)
        if np.any(fc <= 0):
            raise ValueError("fc must be > 0 Hz")
        ratio = 2.0 ** (1.0 / 24.0)
        bw = fc * (ratio - 1.0 / ratio)
        return float(bw[0]) if scalar_input else bw


scale = SemitoneScale()
