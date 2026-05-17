import numpy as np

from .base import ScaleBase


class GreenwoodScale(ScaleBase):
    """Greenwood place-frequency scale (human defaults)."""

    def from_freq(self, frequency, A: float = 165.4, a: float = 2.1, k: float = 0.88):
        r"""Frequency to Greenwood place conversion.

        Converts frequency in Hz to Greenwood place value ``x`` using:

        .. math:: f = A(10^{ax} - k)

        Rearranged for ``x``:

        .. math:: x = \frac{\log_{10}(f/A + k)}{a}

        Parameters
        ----------
        frequency : scalar or ndarray
            Frequency in Hz. Values must be non-negative.
        A : float, optional
            Greenwood scaling constant. Default is ``165.4``.
        a : float, optional
            Greenwood exponential constant. Default is ``2.1``.
        k : float, optional
            Greenwood offset constant. Default is ``0.88``.

        Returns
        -------
        scalar or ndarray
            Greenwood place values.
        """
        frequency = np.asarray(frequency, dtype=float)
        scalar_input = frequency.ndim == 0
        frequency = np.atleast_1d(frequency)
        if np.any(frequency < 0):
            raise ValueError("frequency must be >= 0 Hz")
        x = np.log10(frequency / A + k) / a
        return float(x[0]) if scalar_input else x

    def to_freq(self, scale_value, A: float = 165.4, a: float = 2.1, k: float = 0.88):
        r"""Greenwood place to frequency conversion.

        Converts Greenwood place value ``x`` to frequency in Hz:

        .. math:: f = A(10^{ax} - k)

        Parameters
        ----------
        scale_value : scalar or ndarray
            Greenwood place values.
        A : float, optional
            Greenwood scaling constant. Default is ``165.4``.
        a : float, optional
            Greenwood exponential constant. Default is ``2.1``.
        k : float, optional
            Greenwood offset constant. Default is ``0.88``.

        Returns
        -------
        scalar or ndarray
            Frequencies in Hz.
        """
        scale_value = np.asarray(scale_value, dtype=float)
        scalar_input = scale_value.ndim == 0
        scale_value = np.atleast_1d(scale_value)
        freq = A * (10.0 ** (a * scale_value) - k)
        return float(freq[0]) if scalar_input else freq

    def get_bw(self, fc, A: float = 165.4, a: float = 2.1, k: float = 0.88):
        r"""Approximate bandwidth for a 1-place-unit Greenwood interval.

        Calculates the frequency bandwidth in Hz corresponding to a
        1-place-unit interval centered at ``fc`` in Greenwood space.

        Parameters
        ----------
        fc : scalar or ndarray
            Center frequency in Hz. Values must be non-negative.
        A : float, optional
            Greenwood scaling constant. Default is ``165.4``.
        a : float, optional
            Greenwood exponential constant. Default is ``2.1``.
        k : float, optional
            Greenwood offset constant. Default is ``0.88``.

        Returns
        -------
        scalar or ndarray
            Bandwidth in Hz for a 1-place-unit Greenwood interval.
        """
        fc = np.asarray(fc, dtype=float)
        scalar_input = fc.ndim == 0
        fc = np.atleast_1d(fc)
        if np.any(fc < 0):
            raise ValueError("fc must be >= 0 Hz")
        x_center = self.from_freq(fc, A=A, a=a, k=k)
        upper = self.to_freq(x_center + 0.5, A=A, a=a, k=k)
        lower = self.to_freq(x_center - 0.5, A=A, a=a, k=k)
        bw = upper - lower
        return float(bw[0]) if scalar_input else bw


scale = GreenwoodScale()
