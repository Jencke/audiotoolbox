import audiotools as audio
import audiotools.filter as filter
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

# def test_gauss():
# import matplotlib.pyplot as plt

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
