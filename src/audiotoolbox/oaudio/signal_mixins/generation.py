from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from ... import audiotoolbox as audio

if TYPE_CHECKING:
    from ..signal import Signal


class GenerationMixin:
    """Mixin for signal generation methods."""

    def add_tone(self, frequency, amplitude=1, start_phase=0):
        r"""Add a cosine to the signal.

        This function will add a pure tone to the current
        waveform. following the equation:

        .. math:: x = x + cos(2\pi f t + \phi_0)

        where :math:`x` is the waveform, :math:`f` is the frequency,
        :math:`t` is the time and :math:`\phi_0` the starting phase.
        The first evulated timepoint is 0.

        Parameters
        ----------
        frequency : scalar
            The tone frequency in Hz.
        amplitude : scalar, optional
            The amplitude of the cosine. (default = 1)
        start_phase : scalar, optional
            The starting phase of the cosine. (default = 0)

        Returns
        -------
        Returns itself : Signal

        See Also
        --------
        audiotoolbox.generate_tone

        """
        wv = audio.generate_tone(self.duration, frequency, self.fs, start_phase)

        # If multiple channels are defined, stack them.
        # if self.n_channels > 1:
        #     wv = np.tile(wv, [self.n_channels, 1]).T
        self[:] = (self.T + amplitude * wv.T).T

        return self

    def add_noise(self, ntype="white", variance=1, seed=None):
        r"""Add uncorrelated noise to the signal.

        add gaussian noise with a defined variance and different
        spectral shapes. The noise is generated in the frequency domain
        using the gaussian pseudorandom generator ``numpy.random.randn``.
        The real and imaginarny part of each frequency component is set
        using the psudorandom generator. Each frequency bin is then
        weighted dependent on the spectral shape. The resulting spektrum
        is then transformed into the time domain using ``numpy.fft.ifft``

        Weighting functions:

        - white: :math:`w(f) = 1`
        - pink: :math:`w(f) = \frac{1}{\sqrt{f}}`
        - brown: :math:`w(f) = \frac{1}{f}`

        Parameters
        ----------
        ntype : {'white', 'pink', 'brown'}
            spectral shape of the noise
        variance : scalar, optional
            The Variance of the noise
        seed : int or 1-d array_like, optional
            Seed for `RandomState`.
            Must be convertible to 32 bit unsigned integers.

        Returns
        -------
        Returns itself : Signal

        See Also
        --------
        audiotoolbox.generate_noise
        audiotoolbox.generate_uncorr_noise
        audiotoolbox.Signal.add_uncorr_noise
        """
        noise = audio.generate_noise(
            self.duration, self.fs, ntype=ntype, n_channels=1, seed=seed
        )

        self[:] = (self.T + noise.T * np.sqrt(variance)).T
        return self

    def add_uncorr_noise(
        self,
        corr=0,
        variance=1,
        ntype="white",
        seed=None,
        bandpass=None,
        highpass=None,
        lowpass=None,
    ):
        r"""Add partly uncorrelated noise.

        This function adds partly uncorrelated noise using the N+1
        generator method.

        To generate N partly uncorrelated noises with a desired
        correlation coefficent of $\rho$, the algoritm first generates N+1
        noise tokens which are then orthogonalized using the Gram-Schmidt
        process (as implementd in numpy.linalg.qr). The N+1 th noise token
        is then mixed with the remaining noise tokens using the equation

        .. math:: X_{\rho,n} = X_{N+1}  \sqrt{\rho} + X_n \beta \sqrt{1 - \rho}

        where :math:`X_{\rho,n}` is the nth output and noise,
        :math:`X_{n}` the nth indipendent noise and :math:`X_{N=1}` is the
        common noise.

        for two noise tokens, this is identical to the assymetric
        three-generator method described in [1]_

        Parameters
        ----------
        corr : int, optional
            Desired correlation of the noise tokens, (default=0)
        variance : scalar, optional
            The desired variance of the noise, (default=1)
        ntype : {'white', 'pink', 'brown'}
            spectral shape of the noise
        seed : int or 1-d array_like, optional
            Seed for `RandomState`.
            Must be convertible to 32 bit unsigned integers.
        bandpass : dict, optional
            Parameters for an bandpass filter, these are passed as arguments to
            the audiotoolbox.filter.bandpass function
        lowpass : dict, optional
            Parameters for an lowpass filter, these are passed as arguments to
            the audiotoolbox.filter.lowpass function
        highpass : dict, optional
            Parameters for an highpass filter, these are passed as arguments to
            the audiotoolbox.filter.highpass function

        Returns
        -------
        Returns itself : Signal

        See Also
        --------
        audiotoolbox.generate_noise
        audiotoolbox.generate_uncorr_noise
        audiotoolbox.Signal.add_noise

        References
        ----------
        .. [1] Hartmann, W. M., & Cho, Y. J. (2011). Generating partially
        correlated noise—a comparison of methods. The Journal of the
        Acoustical Society of America, 130(1),
        292–301. http://dx.doi.org/10.1121/1.3596475

        """
        noise = audio.generate_uncorr_noise(
            duration=self.duration,
            fs=self.fs,
            n_channels=self.n_channels,
            ntype=ntype,
            corr=corr,
            seed=seed,
            bandpass=bandpass,
            highpass=highpass,
            lowpass=lowpass,
        )

        self += noise * np.sqrt(variance)

        return self
