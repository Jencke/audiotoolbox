import audiotools as audio
import audiotools.filter as filter
import audiotools.filter.gammatone_filt as gt
import numpy as np
import numpy.testing as testing
from scipy.stats import norm
import pytest

def test_brickwall():
    duration = 500e-3
    fs = 100e3
    noise = audio.generate_noise(duration, fs)

    flow = 200
    fhigh = 400
    out = filter.brickwall(noise, fs, 200, 400)
    spec = np.abs(np.fft.fft(out))
    freqs = np.fft.fftfreq(len(spec), 1. / fs)
    passband = (np.abs(freqs) >= flow) & (np.abs(freqs) <= fhigh)
    non_zero = ~np.isclose(spec, 0)

    assert np.array_equal(non_zero, passband)


    flow = 900
    fhigh = 1130
    out = filter.brickwall(noise, fs, flow, fhigh)
    spec = np.abs(np.fft.fft(out))
    freqs = np.fft.fftfreq(len(spec), 1. / fs)
    passband = (np.abs(freqs) >= flow) & (np.abs(freqs) <= fhigh)
    non_zero = ~np.isclose(spec, 0)

    assert np.array_equal(non_zero, passband)


def test_gauss():
    duration = 500e-3
    fs = 100e3

    noise = audio.generate_noise(duration, fs)

    flow = 500
    fhigh = 1000
    spec = np.zeros_like(noise)
    n_rep = 100
    noise = audio.generate_noise(duration, fs)
    out = filter.gauss(noise, fs, flow, fhigh)

    freqs = np.fft.fftfreq(len(spec), 1. / fs)
    sort_idx = freqs.argsort()
    freqs = freqs[sort_idx]
    mag_spec = np.abs(np.fft.fft(noise))[sort_idx]
    filt_mag_spec = np.abs(np.fft.fft(out))[sort_idx]
    mask = np.where(mag_spec != 0)[0]
    norm_spec = filt_mag_spec[mask] / mag_spec[mask]
    np.warnings.filterwarnings('ignore') # supress divide by zero warning
    lnorm_spec = 20 * np.log10(norm_spec)
    np.warnings.filterwarnings('default')

    assert np.round(lnorm_spec[np.where(freqs[mask] == flow)], 1) == -3
    assert np.round(lnorm_spec[np.where(freqs[mask] == fhigh)], 1) == -3



def test_gammatone():
    # Check that gammatone function is the same as the individual one
    b, a = gt.design_gammatone(500, 75, 48000)
    noise = audio.generate_noise(100e-3, 48000)
    out, states = gt.gammatonefos_apply(noise, b, a, 4)
    out2 = filter.gammatone(noise, 48000, 500, 75)
    # testing.assert_equal(out, out2)

def test_gammatone_coefficients():
    # Compare with results from AMT toolbox
    b, a = gt.design_gammatone(500, 75, 48000, attenuation_db=-3)
    amt_a = 0.98664066847502018831050918379333 + 0.064667845966194278939376260950667j
    amt_b = 0.000000031948804250169196536011229472021

    testing.assert_almost_equal(b, amt_b)
    testing.assert_almost_equal(a[1].imag, -amt_a.imag)
    testing.assert_almost_equal(a[1].real, -amt_a.real)

def test_gammatonefos_apply():
    # Check amplitude with on frequency tone
    b, a = gt.design_gammatone(500, 75, 48000)
    tone = audio.generate_tone(500, 100e-3, 48000)
    out, states = gt.gammatonefos_apply(tone, b, a, 4)
    assert (out.real[3000:].max() - 1) <= 5e-5
    assert (out.real[3000:].min() + 1) <= 5e-5

    # Check magnitude with tone at corner frequency
    b, a = gt.design_gammatone(500, 75, 48000)
    tone = audio.generate_tone(500 - 75 / 2, 100e-3, 48000)
    out, states = gt.gammatonefos_apply(tone, b, a, 4)
    # max should be - 3dB
    assert (20*np.log10(out.real[3000:].max()) + 3) < 0.5e-3


    # Check magnitude with tone at corner frequency
    b, a = gt.design_gammatone(500, 200, 48000)
    tone = audio.generate_tone(500 - 200 / 2, 100e-3, 48000)
    out, states = gt.gammatonefos_apply(tone, b, a, 4)
    # max should be - 3dB
    assert (20*np.log10(out.real[3000:].max()) + 3) < 0.05
