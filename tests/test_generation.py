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
