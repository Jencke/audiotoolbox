import numpy as np

from .base import ScaleBase

class BarkScale(ScaleBase):
    """Object-oriented Bark scale API."""

    _BARK_LIMITS = [
        20,
        100,
        200,
        300,
        400,
        510,
        630,
        770,
        920,
        1080,
        1270,
        1480,
        1720,
        2000,
        2320,
        2700,
        3150,
        3700,
        4400,
        5300,
        6400,
        7700,
        9500,
        12000,
        15500,
    ]

    def get_bark_limits(self):
        r"""Limits of the Bark scale.

        Returns the limit of the Bark scale as defined in [1]_.

        Returns
        -------
        list : Limits of the Bark scale

        References
        ----------
        .. [1] Zwicker, E. (1961). Subdivision of the audible frequency range
            into critical bands (frequenzgruppen). The Journal of the
            Acoustical Society of America, 33(2), 248-248.
            http://dx.doi.org/10.1121/1.1908630
        """
        return list(self._BARK_LIMITS)

    def from_freq(self, frequency, use_table=False):
        r"""Frequency to Bark conversion.

        Converts a given frequency in Hz into the Bark scale using the
        equation by [Traunmueller1990]_ or the original table by
        [Zwicker1961]_.

        Parameters
        ----------
        frequency: scalar or ndarray
            The frequency in Hz. Value has to be between 20 and 15500 Hz.
        use_table: bool, optional
            If True, the original table by [Zwicker1961]_ instead of the
            equation by [Traunmueller1990]_ is used. This also results in the
            critical-band values being returned as integers.

        Returns
        -------
        scalar or ndarray : The critical-band value in Bark.

        References
        ----------
        .. [Zwicker1961] Zwicker, E. (1961). Subdivision of the audible
            frequency range into critical bands (frequenzgruppen). The Journal
            of the Acoustical Society of America, 33(2), 248-248.
            http://dx.doi.org/10.1121/1.19086f30

        .. [Traunmueller1990] Traunmueller, H. (1990). Analytical expressions
            for the tonotopic sensory scale. The Journal of the Acoustical
            Society of America, 88(1), 97-100.
            http://dx.doi.org/10.1121/1.399849
        """
        frequency = np.asarray(frequency, dtype=float)
        scalar_input = frequency.ndim == 0
        frequency = np.atleast_1d(frequency)

        if not np.all(frequency >= 20):
            raise ValueError("frequency must be >= 20 Hz")
        if not np.all(frequency < 15500):
            raise ValueError("frequency must be < 15500 Hz")

        if use_table:
            bark_table = np.array(self.get_bark_limits())
            scale_limits = zip(bark_table[:-1], bark_table[1:])
            i = 0
            cb_val = np.zeros(len(frequency))
            for lower, upper in scale_limits:
                in_border = (frequency >= lower) & (frequency < upper)
                cb_val[in_border] = i
                i += 1
            return cb_val[0] if scalar_input else cb_val

        cb_val = (26.81 * frequency / (1960 + frequency)) - 0.53
        if np.min(cb_val) < 2.0:
            cb_val[cb_val < 2.0] += 0.15 * (2 - cb_val[cb_val < 2.0])
        if np.max(cb_val) > 20.1:
            cb_val[cb_val > 20.1] += 0.22 * (cb_val[cb_val > 20.1] - 20.1)
        return cb_val[0] if scalar_input else cb_val

    def to_freq(self, scale_value):
        r"""Bark to frequency conversion.

        Converts a given value on the Bark scale into frequency using the
        equation by [1]_.

        Parameters
        ----------
        scale_value: scalar or ndarray
            The Bark values.

        Returns
        -------
        scalar or ndarray: The frequency in Hz.

        References
        ----------
        .. [1] Traunmueller, H. (1990). Analytical expressions for the
            tonotopic sensory scale. The Journal of the Acoustical Society of
            America, 88(1), 97-100. http://dx.doi.org/10.1121/1.399849
        """
        bark = np.asarray(scale_value, dtype=float)
        scalar_input = bark.ndim == 0
        bark = np.atleast_1d(bark).copy()

        bark[bark < 2.0] = (bark[bark < 2.0] - 0.3) / 0.85
        bark[bark > 20.1] = (bark[bark > 20.1] + 4.422) / 1.22
        f = 1960 * (bark + 0.53) / (26.28 - bark)
        return f[0] if scalar_input else f

    def get_bw(self, fc):
        r"""Calculate critical bandwidth.

        Returns the critical bandwidth following [Zwicker1980]_.

        Parameters
        ----------
        fc : float or ndarray
            Center frequency in Hz.

        Returns
        -------
        float or ndarray
            The critical bandwidth in Hz.

        References
        ----------
        .. [Zwicker1980] Zwicker, E., & Terhardt, E. (1980). Analytical
            expressions for critical-band rate and critical bandwidth as a
            function of frequency. The Journal of the Acoustical Society of
            America, 68(5), 1523-1525.
        """
        fc = np.asarray(fc, dtype=float)
        scalar_input = fc.ndim == 0
        fc = np.atleast_1d(fc)
        bw = 25 + 75 * (1 + 1.4 * (fc / 1000) ** 2) ** 0.69
        return float(bw[0]) if scalar_input else bw


scale = BarkScale()
