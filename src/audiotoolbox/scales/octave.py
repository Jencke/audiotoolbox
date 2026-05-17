import numpy as np
from numbers import Integral

from .base import ScaleBase


class OctaveScale(ScaleBase):
    """Object-oriented fractional-octave scale API."""

    @staticmethod
    def _get_base(oct_fraction: int, base_system: int):
        if not isinstance(oct_fraction, Integral) or oct_fraction <= 0:
            raise ValueError("oct_fraction must be a positive integer")
        b = oct_fraction
        if base_system == 10:
            gbase = 10 ** (3 / 10)
        elif base_system == 2:
            gbase = 2
        else:
            raise ValueError("base_system must be 2 or 10")
        return b, gbase

    def from_freq(self, frequency, oct_fraction: int = 3, base_system: int = 2):
        """Frequency to octave bandnumber conversion.

        Scales are normalized so that band 1000 Hz is band 30.

        Parameters
        ----------
        frequency: scalar or ndarray
            The frequency in Hz.
        oct_fraction: int
            The fractional octave scale to use. e.g. 3 for 1/3-octave bands.
            Default is 3.
        base_system: {2, 10}
            The base system used for calculation. Default is 2.

        Returns
        -------
        scalar or ndarray
            The octave band number.
        """
        frequency = np.asarray(frequency, dtype=float)
        scalar_input = frequency.ndim == 0
        frequency = np.atleast_1d(frequency)
        b, gbase = self._get_base(oct_fraction, base_system)
        if b % 2:
            band_nr = np.log(frequency / 1000) / np.log(gbase) * b + 30
        else:
            band_nr = 0.5 * (np.log(frequency / 1000) / np.log(gbase) * 2 * b + 59)
        return band_nr[0] if scalar_input else band_nr

    def to_freq(self, scale_value, oct_fraction: int = 3, base_system: int = 2):
        """Octave bandnumber to frequency conversion.

        Converts a (fractional) octave band number to frequency. This can
        either use a base-2 or base-10 system.

        Parameters
        ----------
        scale_value: scalar or ndarray
            The octave band number.
        oct_fraction: int
            The fractional octave scale to use. e.g. 3 for 1/3-octave bands.
            Default is 3.
        base_system: {2, 10}
            The base system used for calculation. Default is 2.

        Returns
        -------
        scalar or ndarray
            The frequencies in Hz.
        """
        scale_value = np.asarray(scale_value, dtype=float)
        scalar_input = scale_value.ndim == 0
        scale_value = np.atleast_1d(scale_value)
        b, gbase = self._get_base(oct_fraction, base_system)
        if b % 2:
            freq = gbase ** ((scale_value - 30.0) / b) * 1e3
        else:
            freq = gbase ** ((2 * scale_value - 59.0) / (2 * b)) * 1e3
        return freq[0] if scalar_input else freq

    def calc_bw(self, fc, oct_fraction: int = 3, base_system: int = 2):
        """Fractional-octave bandwidth around center frequency.

        The bandwidth is computed as upper minus lower band edge with
        symmetric spacing around the center frequency.

        Parameters
        ----------
        fc : float or ndarray
            Center frequency in Hz.
        oct_fraction: int
            The fractional octave scale to use. e.g. 3 for 1/3-octave bands.
            Default is 3.
        base_system: {2, 10}
            The base system used for calculation. Default is 2.

        Returns
        -------
        float or ndarray
            The fractional-octave bandwidth in Hz.
        """
        b, gbase = self._get_base(oct_fraction, base_system)
        ratio = gbase ** (1 / b)
        edge_factor = np.sqrt(ratio)
        fc = np.asarray(fc, dtype=float)
        scalar_input = fc.ndim == 0
        fc = np.atleast_1d(fc)
        bw = fc * (edge_factor - 1 / edge_factor)
        return float(bw[0]) if scalar_input else bw


scale = OctaveScale()
