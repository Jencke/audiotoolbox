import numpy as np
from numpy import testing
import pytest
import audiotoolbox as audio
 

def test_add_noise_basic():
    duration = 1
    fs = 100e3

    # noise = audio.add_noise(duration, fs)
    noise = audio.Signal(1, duration, fs).add_noise()
    assert len(noise) == audio.nsamples(duration, fs)
    assert np.ndim(noise) == 2
    # Test for whole spectrum
    spec = np.fft.fft(noise)
    assert np.all(~np.isclose(np.abs(spec)[1:], 0))
    testing.assert_allclose(np.abs(spec[0]), 0, atol=5)
    testing.assert_almost_equal(np.var(noise), 1)


def test_add_noise_seed():
    duration = 1
    fs = 100e3

    noise1 = audio.Signal(1, duration, fs).add_noise(seed=1)
    noise2 = audio.Signal(1, duration, fs).add_noise(seed=1)
    noise3 = audio.Signal(1, duration, fs).add_noise(seed=2)
    testing.assert_equal(noise1, noise2)
    assert ~np.all(noise1 == noise3)


def test_add_noise_multichannel():
    # test multichannel
    noise = audio.Signal((2, 3), 1, 10).add_noise(ntype="pink")
    assert noise.shape == (10, 2, 3)
    testing.assert_array_equal(noise[:, 0, :], noise[:, 1, :])
    testing.assert_array_equal(noise[:, :, 0], noise[:, :, 1])


def test_add_noise_variance():
    sig = audio.Signal((2, 2), 1, 48000).add_noise(variance=2)
    assert np.var(sig.ch[0]) == np.var(sig.ch[1])
    testing.assert_almost_equal(np.var(sig), 2)


@pytest.mark.parametrize("ntype", ["white", "pink", "brown"])
def test_add_noise_adds_to_existing_signal(ntype):
    # Regression: add_noise must *add* to the signal, not overwrite it.
    # Previously the white-noise branch replaced the signal content.
    offset = 5.0
    sig = audio.Signal(1, 0.1, 48000)
    sig[:] = offset
    sig.add_noise(ntype=ntype, seed=0)
    # the generated noise is zero-mean, so the pre-existing DC offset survives
    testing.assert_allclose(float(sig.mean()), offset, atol=1e-9)


def _excess_kurtosis(x):
    x = np.asarray(x).ravel()
    return np.mean((x - x.mean()) ** 4) / x.var() ** 2 - 3


def _skewness(x):
    x = np.asarray(x).ravel()
    return np.mean((x - x.mean()) ** 3) / x.var() ** 1.5


def _power_law_slope(sig, fs, fmin=50, fmax=5000, nbins=25):
    """Slope of log10(power) vs log10(frequency), log-binned to tame variance."""
    spec = np.abs(np.fft.rfft(np.asarray(sig).ravel())) ** 2
    freq = np.fft.rfftfreq(sig.n_samples, 1 / fs)
    mask = (freq >= fmin) & (freq <= fmax)
    lf, lp = np.log10(freq[mask]), np.log10(spec[mask])
    bins = np.linspace(lf.min(), lf.max(), nbins)
    idx = np.digitize(lf, bins)
    bf = np.array([lf[idx == i].mean() for i in range(1, len(bins)) if np.any(idx == i)])
    bp = np.array([lp[idx == i].mean() for i in range(1, len(bins)) if np.any(idx == i)])
    return np.polyfit(bf, bp, 1)[0]


def test_add_noise_is_gaussian():
    # white noise should be a zero-mean gaussian process
    sig = audio.Signal(1, 4, 48000).add_noise("white", seed=1)
    assert abs(_excess_kurtosis(sig)) < 0.1
    assert abs(_skewness(sig)) < 0.05


def test_add_noise_variance_scaling():
    for variance in (0.5, 2.0, 5.0):
        sig = audio.Signal(1, 4, 48000).add_noise("white", variance=variance, seed=1)
        testing.assert_allclose(float(sig.var()), variance, rtol=1e-2)


@pytest.mark.parametrize(
    ("ntype", "expected_slope"),
    [("white", 0.0), ("pink", -1.0), ("brown", -2.0)],
)
def test_add_noise_spectral_shape(ntype, expected_slope):
    # The documented weighting functions imply a power spectrum of
    # white: f^0, pink: 1/f, brown: 1/f^2.
    fs = 48000
    sig = audio.Signal(1, 4, fs).add_noise(ntype, seed=2)
    slope = _power_law_slope(sig, fs)
    assert abs(slope - expected_slope) < 0.1, (
        f"{ntype} noise slope {slope:.3f}, expected {expected_slope}"
    )


def test_add_noise_invalid_ntype_raises():
    with pytest.raises(ValueError):
        audio.Signal(1, 0.1, 48000).add_noise("blue")


def test_add_uncorr_noise_basic():
    fs = 48000
    sig = audio.Signal(5, 1, fs).add_uncorr_noise()
    # lower trianglular matrix should be  0
    testing.assert_almost_equal(np.tril(np.cov(sig.T), -1), 0)

    # Multidimensional case
    sig = audio.Signal((2, 2), 1, fs).add_uncorr_noise()
    assert sig.n_channels == (2, 2)

    fs = 48000
    sig = audio.Signal(2, 1, fs).add_uncorr_noise()
    # lower trianglular matrix should be  0
    testing.assert_almost_equal(np.tril(np.cov(sig.T), -1), 0)


def test_add_uncorr_noise_seed_produces_independent_noise():
    # Regression: add_noise reseeds the global RNG with the same value for
    # every channel, so with a fixed seed all N+1 noise tokens become
    # identical. The QR step then turns all but one output channel into a
    # structured (non-noise) basis vector instead of independent gaussian
    # noise -- visible as an enormous kurtosis spike.
    fs = 48000
    sig = audio.Signal(4, 1, fs).add_uncorr_noise(corr=0, seed=1)
    arr = np.asarray(sig)

    # excess kurtosis of gaussian noise is ~0; a degenerate spike channel has
    # an enormous kurtosis.
    excess_kurtosis = (
        np.mean((arr - arr.mean(axis=0)) ** 4, axis=0) / arr.var(axis=0) ** 2 - 3
    )
    assert np.all(np.abs(excess_kurtosis) < 1), (
        f"channels are not all gaussian noise; excess kurtosis: {excess_kurtosis}"
    )

    # the fix must keep seeded generation reproducible
    sig2 = audio.Signal(4, 1, fs).add_uncorr_noise(corr=0, seed=1)
    testing.assert_array_equal(arr, np.asarray(sig2))


def test_add_uncorr_noise_negative_corr_two_channels():
    fs = 100e3
    sig = audio.Signal(2, 1, fs).add_uncorr_noise(corr=-0.5, seed=1)
    cc = np.corrcoef(np.asarray(sig).T)[0, 1]
    testing.assert_almost_equal(cc, -0.5, decimal=6)
    # variance is preserved
    testing.assert_almost_equal(np.var(np.asarray(sig), axis=0), 1)


def test_add_uncorr_noise_negative_corr_multichannel_warns():
    fs = 100e3
    with pytest.warns(UserWarning):
        sig = audio.Signal(4, 1, fs).add_uncorr_noise(corr=-0.5, seed=1)
    # falls back to the positive magnitude
    cv = np.corrcoef(np.asarray(sig).T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5)


def test_generate_uncorr_bandpass():
    duration = 1
    fs = 100000
    fc = 300
    bw = 200
    bandpass = {"fc": fc, "bw": bw, "filter_type": "brickwall"}
    noise = audio.Signal(2, duration, fs).add_uncorr_noise(corr=0.5, bandpass=bandpass)
    # noise = audio.generate_uncorr_noise(duration, fs, 2, corr=0.5, bandpass=bandpass)
    flow = fc - bw / 2
    fhigh = fc + bw / 2
    spec = np.abs(np.fft.fft(noise, axis=0))
    freqs = np.fft.fftfreq(len(spec), 1.0 / fs)
    passband = (np.abs(freqs) >= flow) & (np.abs(freqs) <= fhigh)
    non_zero = ~np.isclose(spec, 0)
    assert np.array_equal(non_zero[:, 0], passband)
    assert np.array_equal(non_zero[:, 1], passband)

    # test coherence value
    bandpass = {"fc": fc, "bw": bw, "filter_type": "brickwall"}
    noise = audio.Signal(4, duration, fs).add_uncorr_noise(corr=0.5, bandpass=bandpass)
    cv = np.corrcoef(noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5)

    bandpass = {"fc": fc, "bw": bw, "filter_type": "butter"}
    noise = audio.Signal(4, duration, fs).add_uncorr_noise(corr=0.5, bandpass=bandpass)
    cv = np.corrcoef(noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5, decimal=6)

    bandpass = {"fc": fc, "bw": bw, "filter_type": "gammatone"}
    noise = audio.Signal(4, duration, fs).add_uncorr_noise(corr=0.5, bandpass=bandpass)
    cv = np.corrcoef(noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5, decimal=5)

    fcut = 500
    lowpass = {"f_cut": fcut, "filter_type": "brickwall"}
    noise = audio.Signal(4, duration, fs).add_uncorr_noise(corr=0.5, lowpass=lowpass)
    cv = np.corrcoef(noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5, decimal=5)


def test_add_uncorr_noise_corr():
    duration = 1
    fs = 100e3
    # noise = audio.generate_uncorr_noise(duration, fs, n_channels=2)
    noise = audio.Signal(2, duration, fs).add_uncorr_noise()
    noise1 = noise[:, 0]
    noise2 = noise[:, 1]
    # Test equal Power assumption
    testing.assert_almost_equal(noise1.var(), noise2.var())


def test_add_uncorr_noise_multichannel():
    # Test multichannel
    res_noise = audio.Signal(100, 1, 48000).add_uncorr_noise(corr=0)
    cv = np.corrcoef(res_noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0)

    # Test multichannel
    res_noise = audio.Signal(3, 1, 48000).add_uncorr_noise(corr=0.5)
    cv = np.corrcoef(res_noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5)

    # Test multichannel
    res_noise = audio.Signal(3, 1, 48000).add_uncorr_noise(corr=0.5, ntype="pink")
    cv = np.corrcoef(res_noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5)


def test_add_uncorr_noise_variance():
    duration = 1
    fs = 100e3
    # Test vor variance = 1
    noise = audio.Signal(2, duration, fs).add_uncorr_noise(corr=0.5)
    testing.assert_almost_equal(noise.var(axis=0), 1)


def test_add_uncorr_noise_mult_dim():
    duration = 1
    fs = 48000
    # Test multiple dimensions:
    noise = audio.Signal((2, 3, 4), duration, fs).add_uncorr_noise(corr=0.5)
    assert noise.shape[1:] == (2, 3, 4)
    noise = noise.reshape([len(noise), 2 * 3 * 4])
    cv = np.corrcoef(noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5)
