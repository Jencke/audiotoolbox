import audiotools as audio
import audiotools.filter as filter
import numpy as np
import numpy.testing as testing
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
