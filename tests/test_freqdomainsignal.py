from audiotools.oaudio import FrequencyDomainSignal, Signal
import audiotools as audio
import numpy as np
import numpy.testing as testing


def test_basics():
    sig = Signal(1, 1, 100)
    assert sig.n_channels == 1
    sig = sig.to_freqdomain()
    assert sig.n_channels == 1
    assert sig.freq.min() == -50
    assert sig.freq.max() == 49
    assert sig.omega.max() == 49 * np.pi * 2

    sig = FrequencyDomainSignal(1, 1, 100)
    assert sig.dtype == np.dtype("complex128")
    assert sig.n_channels == 1
    assert sig.freq.min() == -50
    assert sig.freq.max() == 49
    assert sig.omega.max() == 49 * np.pi * 2


def test_slice():
    sig = FrequencyDomainSignal(2, 1, 100)
    assert sig._fs == 100
    sig2 = sig[:, 0]
    assert sig2._fs == 100


def test_phaseshift():
    # phase shifting a full period
    sig = audio.Signal(2, 1, 48000).add_tone(100)
    sig[:, 1] = sig[:, 1].to_freqdomain().phase_shift(2 * np.pi).to_timedomain()
    testing.assert_almost_equal(sig[:, 1], sig[:, 0])

    # phase shifting half a period
    sig = audio.Signal(2, 1, 48000)
    sig[:, 0].add_tone(100, start_phase=-0.5 * np.pi)
    sig[:, 1].add_tone(100)
    sig[:, 1] = sig[:, 1].to_freqdomain().phase_shift(0.5 * np.pi).to_timedomain()
    testing.assert_almost_equal(sig[:, 1], sig[:, 0])

    # phase shifting half a period
    sig = audio.Signal((2, 2), 1, 48000)
    sig[:, :, 0].add_tone(100, start_phase=-0.5 * np.pi)
    sig[:, :, 1].add_tone(100)
    sig[:, :, 1] = sig[:, :, 1].to_freqdomain().phase_shift(0.5 * np.pi).to_timedomain()
    testing.assert_almost_equal(sig[:, :, 1], sig[:, :, 0])


def test_timeshift():
    # timeshift a tone by one full phase
    sig = audio.Signal(2, 1, 48000).add_tone(100)
    sig[:, 1] = sig[:, 1].to_freqdomain().time_shift(1.0 / 100).to_timedomain()
    testing.assert_almost_equal(sig[:, 1], sig[:, 0])

    shift_samples = 500
    shift_time = shift_samples / 48000
    sig = audio.Signal(2, 1, 48000).add_noise()
    sig[:, 1] = sig[:, 1].to_freqdomain().time_shift(shift_time).to_timedomain()
    testing.assert_almost_equal(sig[shift_samples:, 1], sig[:-shift_samples, 0])

    shift_time = 3.2e-4
    sig = audio.Signal(2, 1.000, 48000).add_noise()
    res = sig[:, 1].to_freqdomain().time_shift(shift_time).to_timedomain()
    assert np.all(np.isreal(res))

    shift_time = 3.2e-4
    sig = audio.Signal((2, 2), 1.000, 48000).add_noise()
    res = sig[:, 1].to_freqdomain().time_shift(shift_time).to_timedomain()
    assert np.all(np.isreal(res))


def test_analytical():
    sig = audio.Signal(2, 1, 48000).add_noise()
    fsig = sig.to_freqdomain()
    fsig.to_analytical()
    # All but the niquist bin should be zero
    testing.assert_array_equal(fsig[fsig.freq < 0][1:], 0)
    asig = fsig.to_timedomain()
    testing.assert_almost_equal(asig.real, sig)

    sig = audio.Signal(1, 0.1, 48000).add_tone(500)
    sig2 = audio.Signal(1, 0.1, 48000).add_tone(500, start_phase=-np.pi / 2)
    fsig = sig.to_freqdomain()
    fsig.to_analytical()
    asig = fsig.to_timedomain()
    testing.assert_almost_equal(asig.imag, sig2)
    testing.assert_almost_equal(asig.real, sig)

    sig = audio.Signal((2, 2), 1, 48000).add_noise()
    fsig = sig.to_freqdomain()
    fsig.to_analytical()
    asig = fsig.to_timedomain()
    testing.assert_almost_equal(asig.real, sig)


def test_freqdomain_multiple_copy():
    sig = audio.Signal((2, 2), 1, 48000).add_noise()
    fsig = sig.to_freqdomain()
    fsig_orig = fsig.copy()
    fsig.to_timedomain()
    np.testing.assert_array_equal(fsig_orig, fsig)
