import numpy as np

from .base import ScaleBase

class ErbScale(ScaleBase):
    """Object-oriented ERB scale API."""

    def from_freq(self, frequency):
        r"""Frequency to number of ERBs conversion.

        Calculates the number of ERBs for a given sound frequency in Hz
        using the equation by [1]_.

        Parameters
        ----------
        frequency: scalar or ndarray
            The frequency in Hz.

        Returns
        -------
        scalar or ndarray
            The number of ERBs corresponding to the frequency.

        References
        ----------
        .. [1] Glasberg, B. R., & Moore, B. C. (1990). Derivation of auditory
            filter shapes from notched-noise data. Hearing Research, 47(1-2),
            103-138.
        """
        frequency = np.asarray(frequency, dtype=float)
        scalar_input = frequency.ndim == 0
        frequency = np.atleast_1d(frequency)
        erb = (1000.0 / (24.7 * 4.37)) * np.log(4.37 * frequency / 1000 + 1)
        return erb[0] if scalar_input else erb

    def to_freq(self, scale_value):
        r"""Number of ERBs to frequency conversion.

        Calculates the frequency from a given number of ERBs using the
        equation by [1]_.

        Parameters
        ----------
        scale_value: scalar or ndarray
            The number of ERBs.

        Returns
        -------
        scalar or ndarray
            The corresponding frequency in Hz.

        References
        ----------
        .. [1] Glasberg, B. R., & Moore, B. C. (1990). Derivation of auditory
            filter shapes from notched-noise data. Hearing Research, 47(1-2),
            103-138.
        """
        scale_value = np.asarray(scale_value, dtype=float)
        scalar_input = scale_value.ndim == 0
        scale_value = np.atleast_1d(scale_value)
        fkhz = (np.exp(scale_value * (24.7 * 4.37) / 1000) - 1) / 4.37
        freq = fkhz * 1000
        return freq[0] if scalar_input else freq

    def calc_bw(self, fc):
        r"""Calculate bandwidth on the ERB scale.

        Returns the equivalent rectangular bandwidth for a given center
        frequency following [Glasberg1990]_.

        Parameters
        ----------
        fc : float or ndarray
            Center frequency in Hz.

        Returns
        -------
        float or ndarray
            The ERB in Hz.

        References
        ----------
        .. [Glasberg1990] Glasberg, B. R., & Moore, B. C. (1990). Derivation
            of auditory filter shapes from notched-noise data. Hearing
            Research, 47(1-2), 103-138.
        """
        fc = np.asarray(fc, dtype=float)
        scalar_input = fc.ndim == 0
        fc = np.atleast_1d(fc)
        bw = 24.7 * (4.37 * (fc / 1000) + 1)
        return float(bw[0]) if scalar_input else bw


scale = ErbScale()
