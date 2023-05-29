import audiotools as audio
import audiotools.filter as filter
import audiotools.filter.gammatone_filt as gt
import numpy as np
import numpy.testing as testing



def test_brickwall():
    # Test Bandpass
    duration = 500e-3
    fs = 100e3
    noise = audio.generate_noise(duration, fs)

    fc = 300
    bw = 200
    flow = fc - bw / 2
    fhigh = fc + bw / 2
    out = filter.brickwall(noise, flow, fhigh, fs)
    spec = np.abs(np.fft.fft(out))
    freqs = np.fft.fftfreq(len(spec), 1. / fs)
    passband = (np.abs(freqs) >= flow) & (np.abs(freqs) <= fhigh)
    non_zero = ~np.isclose(spec, 0)

    assert np.array_equal(non_zero, passband)

    fc = 1015
    bw = 230
    flow = 900
    fhigh = 1130
    out = filter.brickwall(noise, flow, fhigh, fs)
    spec = np.abs(np.fft.fft(out))
    freqs = np.fft.fftfreq(len(spec), 1. / fs)
    passband = (np.abs(freqs) >= flow) & (np.abs(freqs) <= fhigh)
    non_zero = ~np.isclose(spec, 0)

    assert np.array_equal(non_zero, passband)

    # Brickwall Lowpass

    duration = 500e-3
    fs = 100e3
    noise = audio.generate_noise(duration, fs)

    fc = 300
    out = filter.brickwall(noise, None, fc, fs)
    spec = np.abs(np.fft.fft(out))
    freqs = np.fft.fftfreq(len(spec), 1. / fs)

    # check if only frequencies within the passband are non-zero
    passband = ((np.abs(freqs) <= fc) & (freqs != 0))
    non_zero = ~np.isclose(spec, 0)

    assert np.array_equal(non_zero, passband)

    # brickwall_highpass
    duration = 500e-3
    fs = 100e3
    noise = audio.generate_noise(duration, fs)

    fc = 300
    out = filter.brickwall(noise, fc, None, fs)
    spec = np.abs(np.fft.fft(out))
    freqs = np.fft.fftfreq(len(spec), 1. / fs)

    # check if only frequencies within the passband are non-zero
    passband = ((np.abs(freqs) >= fc))
    non_zero = ~np.isclose(spec, 0)

    assert np.array_equal(non_zero, passband)


def test_butterworth_filt():
    # Test Lowpass
    sig = audio.Signal(10, 1, 48000).add_tone(500)
    sig_out = filter.butterworth(sig, None, 500, 48000)
    testing.assert_almost_equal(sig_out[:].std(), 0.5, 3)

    sig = audio.Signal(1, 1, 48000).add_tone(400)
    sig_out = filter.butterworth(sig, None, 500, 48000)
    assert sig_out[:].std() > 0.5

    sig = audio.Signal(1, 1, 48000).add_tone(600)
    sig_out = filter.butterworth(sig, None, 500, 48000)
    assert sig_out[:].std() < 0.5

    # test Highpass
    sig = audio.Signal(1, 1, 48000).add_tone(500)
    sig_out = filter.butterworth(sig, 500, None, 48000)
    testing.assert_almost_equal(sig_out[:].std(), 0.5, 3)

    sig = audio.Signal(1, 1, 48000).add_tone(400)
    sig_out = filter.butterworth(sig, 500, None, 48000)
    assert sig_out[:].std() < 0.5

    sig = audio.Signal(1, 1, 48000).add_tone(600)
    sig_out = filter.butterworth(sig, 500, None, 48000)
    assert sig_out[:].std() > 0.5

    sig = audio.Signal(1, 1, 48000).add_tone(500)
    sig_out = filter.butterworth(sig, 400, 600, 48000)
    testing.assert_almost_equal(sig_out[:].std(), 1 / np.sqrt(2), 3)
    sig = audio.Signal(1, 1, 48000).add_tone(400)
    sig_out = filter.butterworth(sig, 400, 600, 48000)
    testing.assert_almost_equal(sig_out[:].std(), 0.5, 3)
    sig = audio.Signal(1, 1, 48000).add_tone(600)
    sig_out = filter.butterworth(sig, 400, 600, 48000)
    testing.assert_almost_equal(sig_out[:].std(), 0.5, 3)

    sig = audio.Signal((2, 3), 1, 48000).add_tone(500)
    sig_out = filter.butterworth(sig, 400, 600, 48000)
    testing.assert_almost_equal(np.std(sig_out, axis=0), 1 / np.sqrt(2), 3)
    sig = audio.Signal((2, 3), 1, 48000).add_tone(400)
    sig_out = filter.butterworth(sig, 400, 600, 48000)
    testing.assert_almost_equal(np.std(sig_out, axis=0), 0.5, 3)

def test_gammatone():
    # Tone located at center frequency
    sig = audio.Signal(1, 1, 48000).add_tone(500)
    out = filter.gammatone(sig, 500, 75, 48000,
                           order=4, attenuation_db=-3)
    testing.assert_almost_equal(out.real.std(), 1 / np.sqrt(2), 2)

    # Tone located at -3dB frequency
    sig = audio.Signal(2, 1, 48000).add_tone(400)
    out = filter.gammatone(sig, 500, 200, 48000,
                           order=4, attenuation_db=-3)
    testing.assert_almost_equal(out.real.std(), 0.5, 2)

    # Test Multichannel
    sig = audio.Signal((2, 2), 1, 48000).add_tone(400)
    out = filter.gammatone(sig, 500, 200, 48000,
                           order=4, attenuation_db=-3)
    testing.assert_almost_equal(out.real.std(), 0.5, 2)
    assert(sig.shape == out.shape)

def test_gammatone_coefficients():
    # Compare with results from AMT toolbox
    b, a = gt.design_gammatone(500, 75, 48000, attenuation_db=-3)
    amt_a = (0.98664066847502018831050918379333
             + 0.064667845966194278939376260950667j)
    amt_b = 0.000000031948804250169196536011229472021

    testing.assert_almost_equal(b, amt_b)
    testing.assert_almost_equal(a[1].imag, -amt_a.imag)
    testing.assert_almost_equal(a[1].real, -amt_a.real)

def test_gammatonefos_apply():
    # Check amplitude with on frequency tone
    b, a = gt.design_gammatone(500, 75, 48000, attenuation_db=-3)
    tone = audio.generate_tone(100e-3, 500, 48000)
    out, states = gt.gammatonefos_apply(tone, b, a, 4)
    assert (out.real[3000:].max() - 1) <= 5e-5
    assert (out.real[3000:].min() + 1) <= 5e-5

    # Check magnitude with tone at corner frequency
    b, a = gt.design_gammatone(500, 75, 48000, attenuation_db=-3)
    tone = audio.generate_tone(100e-3, 500 - 75 / 2, 48000)
    out, states = gt.gammatonefos_apply(tone, b, a, 4)
    # max should be - 3dB
    assert (20*np.log10(out.real[3000:].max()) + 3) < 0.5e-3

    # Check magnitude with tone at corner frequency
    b, a = gt.design_gammatone(500, 200, 48000, attenuation_db=-3)
    tone = audio.generate_tone(100e-3, 500 - 200 / 2, 48000)
    out, states = gt.gammatonefos_apply(tone, b, a, 4)
    # max should be - 3dB
    assert (20*np.log10(out.real[3000:].max()) + 3) < 0.05

def test_bandpass():
    sig = audio.Signal(1, 1, 48000).add_noise()

    sig_out = audio.filter.bandpass(sig, 500, 100, 'gammatone')
    sig_out2 = audio.filter.gammatone(sig, 500, 100, 48000)
    testing.assert_array_equal(sig_out, sig_out2)

    sig = audio.Signal(1, 1, 48000).add_noise()
    sig_out = audio.filter.bandpass(sig, 500, 100, 'butter')
    sig_out2 = audio.filter.butterworth(sig, 450, 550, 48000)
    testing.assert_array_equal(sig_out, sig_out2)

    sig = audio.Signal(1, 1, 48000).add_noise()
    sig_out = audio.filter.bandpass(sig, 500, 100, 'brickwall')
    sig_out2 = audio.filter.brickwall(sig, 450, 550, 48000)
    testing.assert_array_equal(sig_out, sig_out2)

def test_lowpass():
    sig = audio.Signal(1, 1, 48000).add_noise()

    sig_out = audio.filter.lowpass(sig, 500, 'butter')
    sig_out2 = audio.filter.butterworth(sig, None, 500, 48000)
    testing.assert_array_equal(sig_out, sig_out2)

    sig_out = audio.filter.lowpass(sig, 500, 'brickwall')
    sig_out2 = audio.filter.brickwall(sig, None, 500, 48000)
    testing.assert_array_equal(sig_out, sig_out2)

def test_highpass():
    sig = audio.Signal(1, 1, 48000).add_noise()

    sig_out = audio.filter.highpass(sig, 500, 'butter')
    sig_out2 = audio.filter.butterworth(sig, 500, None,  48000)
    testing.assert_array_equal(sig_out, sig_out2)

    sig_out = audio.filter.highpass(sig, 500, 'brickwall')
    sig_out2 = audio.filter.brickwall(sig, 500, None, 48000)
    testing.assert_array_equal(sig_out, sig_out2)
