from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional, Union
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
            2 * np.pi * frequency[None, :] * self.time[:, None] + start_phase[None, :]
        )
        summed_tones = np.sum(amplitude[None, :] * np.cos(phases), axis=1)

        # Reshape the summed tones vector for broadcasting to all channels
        # e.g., (n_samples,) -> (n_samples, 1, 1) for a 3D signal.
        new_shape = (-1,) + (1,) * (self.ndim - 1)
        tones_to_add = summed_tones.reshape(new_shape)

        self += tones_to_add
        return self

    def add_noise(
        self,
        ntype: Literal["white", "pink", "brown"] = "white",
        variance: float = 1.0,
        seed=None,
    ):
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
        audiotoolbox.Signal.add_uncorr_noise
        """
        np.random.seed(seed)

        # If noise type is white just use the random number generator
        if ntype == "white":
            noise = np.random.randn(self.n_samples)
            noise -= noise.mean(axis=0)
            # normalize variance
            noise /= noise.std(axis=0)
            noise *= np.sqrt(variance)

            new_shape = (self.n_samples,) + (1,) * (self.ndim - 1)
            self[:] = noise.reshape(new_shape)
            return self

        # Otherwise create spectrum
        # Calculate length and number of fft samples
        nfft = audio.nextpower2(self.n_samples)

        df = self.fs / nfft  # Frequency resolution
        nybin = nfft // 2 + 1  # nyquist bin

        lowbin = 1  # no offset start at one
        highbin = nybin

        freqs = np.arange(0, nybin) * df

        # amplitude weighting factor
        f_weights = np.zeros(nfft)
        if ntype == "pink":
            # Power proportinal to 1 / f
            f_weights[lowbin:highbin] = 1.0 / np.sqrt(freqs[lowbin:])
        elif ntype == "brown":
            # Power proportional to 1 / f**2
            f_weights[lowbin:highbin] = 1.0 / freqs[lowbin:]
        else:
            raise (ValueError("ntype not implemented"))

        # generate noise
        a = np.zeros([nfft])
        b = np.zeros([nfft])
        a[lowbin:highbin] = np.random.randn(highbin - lowbin)
        b[lowbin:highbin] = np.random.randn(highbin - lowbin)
        fspec = a + 1j * b

        # Frequency weighting
        fspec *= f_weights

        noise = np.fft.ifft(fspec, axis=0)
        noise = np.real(noise)

        noise = noise[: self.n_samples]
        noise -= noise.mean()

        # Normalize the signal by its rms
        noise /= np.std(noise)
        noise *= np.sqrt(variance)

        new_shape = (self.n_samples,) + (1,) * (self.ndim - 1)
        self += noise.reshape(new_shape)
        return self

    def add_uncorr_noise(
        self,
        corr: float = 0,
        variance: float = 1,
        ntype: Literal["white", "pink", "brown"] = "white",
        seed: Optional[float | None] = None,
        bandpass: Optional[dict] = None,
        highpass: Optional[dict] = None,
        lowpass: Optional[dict] = None,
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
        audiotoolbox.Signal.add_noise

        References
        ----------
        .. [1] Hartmann, W. M., & Cho, Y. J. (2011). Generating partially
        correlated noise—a comparison of methods. The Journal of the
        Acoustical Society of America, 130(1),
        292-301. http://dx.doi.org/10.1121/1.3596475

        """
        if corr < 0:
            Warning(
                ValueError(
                    "Resulting correlations will be positive"
                    + " to gain negative correlations, multiply"
                    + " channel with -1"
                )
            )
        corr = np.abs(corr)
        # if more then one dimension in n_channels
        if np.ndim(self.n_channels) > 0:
            shape = self.n_channels
            n_channels = np.prod(self.n_channels)
        else:
            shape = self.n_channels
            n_channels = self.n_channels

        # correlated noise in multiple channels is generated by using the
        # N+1 generator method

        noise = audio.Signal(n_channels + 1, self.duration, self.fs)
        for ch in range(noise.n_channels):
            noise.ch[ch].add_noise(ntype=ntype, seed=seed)
        noise -= noise.mean(axis=0)

        if bandpass is not None:
            noise = noise.bandpass(**bandpass)
        if lowpass is not None:
            noise = noise.lowpass(**lowpass)
        if highpass is not None:
            noise = noise.highpass(**highpass)

        # normalize variance
        noise /= noise.std(axis=0)

        # Orthogonalize the noise tokens
        Q, R = np.linalg.qr(noise, "reduced")

        # normalizing the individual noise energies somewhat reduces
        # the trial-by-trial variance of correlation values
        Q /= Q.std(axis=0)

        # The common noise component is mixed with each of the independent
        # noise components to reach the desired correlation
        common_noise = Q.ch[-1]
        independent_noise = Q.ch[:-1]
        #
        alpha = np.sqrt(corr)
        beta = np.sqrt(1 - alpha**2)
        res_noise = (common_noise.T * alpha + independent_noise.T * beta).T

        # Again make sure that the output variance is 1
        res_noise /= res_noise.std(axis=0)

        # bring into correct shape
        if np.ndim(shape) > 0:
            full_shape = [len(res_noise), *shape]
            res_noise = res_noise.reshape(full_shape)
        # if really only 1 dimensional, return vector
        elif res_noise.shape[1] == 1:
            res_noise = np.squeeze(res_noise)

        self += res_noise * np.sqrt(variance)

        return self
