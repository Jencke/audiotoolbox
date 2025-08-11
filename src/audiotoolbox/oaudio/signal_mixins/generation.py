from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from ... import audiotoolbox as audio

if TYPE_CHECKING:
    from ..signal import Signal


class GenerationMixin:
    """Mixin for signal generation methods."""

    def add_tone(
        self: "Signal",
        frequency: Union[float, np.ndarray, list],
        amplitude: Union[float, np.ndarray, list] = 1,
        start_phase: Union[float, np.ndarray, list] = 0,
    ) -> "Signal":
        r"""Add one or more cosine tones to the signal.

        This function will add pure tones to the current
        waveform. If multiple frequencies are given (as arrays), their
        waveforms are summed together before being added to the signal.

        .. math:: x_{new} = x_{old} + \sum_{i} A_i \cos(2\pi f_i t + \phi_{0,i})

        Parameters
        ----------
        frequency : float or array-like
            The tone frequency or frequencies in Hz.
        amplitude : float or array-like, optional
            The amplitude of the cosine(s). Must have the same
            length as `frequency` if provided as an array. (default = 1)
        start_phase : float or array-like, optional
            The starting phase of the cosine(s) in radians. Must have
            the same length as `frequency` if provided as an array. (default = 0)

        Returns
        -------
        Signal
            Returns self for method chaining.
        """
        # Ensure inputs are 1D arrays for consistent processing
        frequency = np.atleast_1d(frequency)
        amplitude = np.atleast_1d(amplitude)
        start_phase = np.atleast_1d(start_phase)

        # Validate that inputs are 1D
        if not (frequency.ndim == 1 and amplitude.ndim == 1 and start_phase.ndim == 1):
            raise ValueError(
                "Inputs for frequency, amplitude, and start_phase must be scalars or 1D arrays."
            )

        # Check that if multiple arrays are given, they have the same length
        arrs = [arr for arr in (frequency, amplitude, start_phase) if arr.size > 1]
        if arrs:
            it = iter(arrs)
            the_len = len(next(it))
            if not all(len(l) == the_len for l in it):
                raise ValueError(
                    "When providing arrays, frequency, amplitude, and start_phase must have the same length."
                )

        # Generate all tones and apply amplitude before summing.
        # Broadcasting handles scalar vs. array inputs.
        # `self.time[:, None]` -> shape (n_samples, 1)
        # `frequency[None, :]` -> shape (1, n_freqs)
        # Resulting `phases` shape: (n_samples, n_freqs)
        phases = (
            2 * np.pi * frequency[None, :] * self.time[:, None]
            + start_phase[None, :]
        )
        summed_tones = np.sum(amplitude[None, :] * np.cos(phases), axis=1)

        # Reshape the summed tones vector for broadcasting to all channels
        # e.g., (n_samples,) -> (n_samples, 1, 1) for a 3D signal.
        new_shape = (-1,) + (1,) * (self.ndim - 1)
        tones_to_add = summed_tones.reshape(new_shape)

        self += tones_to_add
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
