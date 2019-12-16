import unittest

from audiotools.oaudio import *
import audiotools as audio
import numpy as np
import numpy.testing as testing
import pytest


class test_oaudio(unittest.TestCase):

    def test_set_waveform(self):
        sig = Signal()
        assert sig.n_channels == 0
        assert sig.n_samples == 0
        assert sig.duration == 0


        # test set single channel signal
        sig = Signal()
        sig.set_waveform(np.zeros(100), 100)
        assert sig.n_channels == 1
        assert sig.n_samples == 100
        assert sig.duration == 1


        # test multiple channel signal
        sig = Signal()
        sig.set_waveform(np.zeros([100, 2]), 100)
        assert sig.n_channels == 2
        assert sig.n_samples == 100
        assert sig.duration == 1

    def test_fs(self):
        sig = Signal()
        assert sig.fs == None
        sig.set_waveform(np.zeros([100, 2]), 100)
        assert sig.fs == 100

        with self.assertRaises(ValueError):
            sig.fs = 200

        sig = Signal()
        with self.assertRaises(ValueError):
            sig.fs = None

    def test_init_signal(self):
        sig = Signal()
        sig.init_signal(1, 100, 1)

        assert sig.fs == 1
        assert sig.duration == 100
        assert sig.n_samples == 100


    def test_time(self):
        sig = Signal()
        sig.init_signal(1, 100, 1)
        time = sig.time

        assert sig.time[0] == 0
        assert np.all(np.diff(time) == 1)
        assert sig.time[-1] == 99

    def test_addtone(self):
        fs = 48000
        duration = 100e-3

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100)
        sig.add_tone(200, start_phase=np.pi)

        test = audio.generate_tone(100, duration, fs)
        test += audio.generate_tone(200, duration, fs, np.pi)

        testing.assert_equal(sig.waveform, test)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100, amplitude=2)

        test = 2 * audio.generate_tone(100, duration, fs)

        testing.assert_equal(sig.waveform, test)

    def test_setdbspl(self):
        fs = 48000
        duration = 100e-3

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).set_dbspl(50)

        test = audio.generate_tone(100, duration, fs)
        test = audio.set_dbspl(test, 50)

        testing.assert_equal(sig.waveform, test)

    def test_calcdbspl(self):
        fs = 48000
        duration = 100e-3

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).set_dbspl(50)
        db = sig.calc_dbspl()

        assert db == 50

    def test_setdbfs_calcdbfs(self):
        fs = 48000
        duration = 100e-3

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).set_dbfs(-5)

        assert(audio.calc_dbfs(sig.waveform) == -5)
        assert(sig.calc_dbfs() == -5)


    def test_zeropad(self):
        fs = 48000
        duration = 100e-3

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).zeropad(number=10)
        wv = sig.waveform
        assert np.all(wv[:10] == wv[-10:])
        assert np.all(wv[:10] == 0)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).zeropad(number=[10, 5])
        wv = sig.waveform
        assert np.all(wv[:10] == 0)
        assert np.all(wv[-5:] == 0)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).zeropad(duration=10e-3)
        n_zeros = audio.nsamples(10e-3, fs)
        wv = sig.waveform
        assert np.all(wv[:n_zeros] == 0)
        assert np.all(wv[-n_zeros:] == 0)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).zeropad(duration=[5e-3, 10e-3])
        n_zeros_s = audio.nsamples(5e-3, fs)
        n_zeros_e = audio.nsamples(10e-3, fs)
        wv = sig.waveform
        assert np.all(wv[:n_zeros_s] == 0)
        assert np.all(wv[-n_zeros_e:] == 0)

    def test_fadewindow(self):
        fs = 48000
        duration = 100e-3

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).fade_window(rise_time=10e-3, type='gauss')
        test = audio.generate_tone(100, duration, fs)
        test *= audio.gaussian_fade_window(test, 10e-3, fs)
        testing.assert_equal(sig.waveform, test)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).fade_window(rise_time=10e-3, type='cos')
        test = audio.generate_tone(100, duration, fs)
        test *= audio.cosine_fade_window(test, 10e-3, fs)
        testing.assert_equal(sig.waveform, test)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).fade_window(rise_time=10e-3, type='cos')
        test = audio.generate_tone(100, duration, fs)
        test *= audio.cosine_fade_window(test, 10e-3, fs)
        testing.assert_equal(sig.waveform, test)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100).fade_window(rise_time=10e-3, type='hann')
        test = audio.generate_tone(100, duration, fs)
        test *= audio.hann_fade_window(test, 10e-3, fs)
        testing.assert_equal(sig.waveform, test)

    def test_add(self):
        fs = 48000
        duration = 100e-3

        # test addition of signal
        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100)
        sig2 = Signal()
        sig2.init_signal(1, duration, fs)
        sig2.add_tone(200)
        sig.add(sig2)
        test = Signal()
        test.init_signal(1, duration, fs)
        test.add_tone(100).add_tone(200)
        testing.assert_equal(sig.waveform, test.waveform)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100)
        sig.add(2).add(1.0)
        test = Signal()
        test.init_signal(1, duration, fs)
        test.add_tone(100)
        testing.assert_almost_equal(sig.waveform, test.waveform + 3.0)

        sig = Signal()
        sig.init_signal(2, duration, fs)
        sig.add_tone(100)
        sig.add(np.array([1, 2]))
        testing.assert_allclose(sig[1].waveform.mean() - sig[0].waveform.mean(), 1)

        sig = Signal()
        sig.init_signal(2, duration, fs)
        sig.add_tone(100)
        sig[1].add(sig[0].waveform)
        testing.assert_allclose(sig[1].waveform /  sig[0].waveform, 2)


    def test_subtract(self):
        fs = 48000
        duration = 100e-3

        # test addition of signal
        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100)
        sig2 = Signal()
        sig2.init_signal(1, duration, fs)
        sig2.add_tone(100)
        sig.subtract(sig2)
        testing.assert_allclose(sig.waveform, 0)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100)
        sig.subtract(2).subtract(1.0)
        test = Signal()
        test.init_signal(1, duration, fs)
        test.add_tone(100)
        testing.assert_almost_equal(sig.waveform, test.waveform - 3.0)

        sig = Signal()
        sig.init_signal(2, duration, fs)
        sig.add_tone(100)
        sig.subtract(np.array([1, 2]))
        testing.assert_allclose(sig[1].waveform.mean() - sig[0].waveform.mean(), -1)

        sig = Signal()
        sig.init_signal(2, duration, fs)
        sig.add_tone(100)
        sig[1].subtract(2 * sig[0].waveform)
        testing.assert_allclose(sig[1].waveform /  sig[0].waveform, -1)


    def test_multiply(self):
        fs = 48000
        duration = 100e-3

        # test addition of signal
        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100)
        sig2 = Signal()
        sig2.init_signal(1, duration, fs)
        sig2.add_tone(100)
        sig.multiply(sig2)
        testing.assert_almost_equal(sig.waveform, sig2.waveform**2)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100)
        sig.multiply(2).multiply(2.1)
        test = Signal()
        test.init_signal(1, duration, fs)
        test.add_tone(100)
        testing.assert_almost_equal(sig.waveform, test.waveform * 2 * 2.1)

        sig = Signal()
        sig.init_signal(2, duration, fs)
        sig.add_tone(100)
        sig.multiply(np.array([1, 2]))
        testing.assert_almost_equal(sig[1].waveform, sig[0].waveform * 2)

        sig = Signal()
        sig.init_signal(2, duration, fs)
        sig.add_tone(100)
        sig[1].multiply(sig[0].waveform)
        testing.assert_almost_equal(sig[1].waveform,  sig[0].waveform**2)

    def test_divide(self):
        fs = 48000
        duration = 100e-3

        # test addition of signal
        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100)
        sig2 = Signal()
        sig2.init_signal(1, duration, fs)
        sig2.add_tone(100)
        sig.divide(sig2)
        testing.assert_allclose(sig.waveform, 1)

        sig = Signal()
        sig.init_signal(1, duration, fs)
        sig.add_tone(100)
        sig.divide(2).divide(2.1)
        test = Signal()
        test.init_signal(1, duration, fs)
        test.add_tone(100)
        testing.assert_almost_equal(sig.waveform, test.waveform / 2 / 2.1)

        sig = Signal()
        sig.init_signal(2, duration, fs)
        sig.add_tone(100)
        sig.divide(np.array([1, 2]))
        testing.assert_almost_equal(sig[1].waveform, sig[0].waveform / 2)

        sig = Signal()
        sig.init_signal(2, duration, fs)
        sig.add_tone(100)
        sig[1].divide(sig[0].waveform)
        testing.assert_allclose(sig[1].waveform, 1)

    def test_mean(self):
        sig = Signal()
        sig.init_signal(2, 100e-3, 100e3)
        sig.add_tone(100).add(np.array([1, 2]))
        mean = sig.mean()
        testing.assert_almost_equal(mean, np.array([1, 2]))

    def test_rms(self):
        sig = Signal()
        sig.init_signal(2, 100e-3, 100e3)
        sig.add_tone(100)
        rms = sig.rms()
        testing.assert_allclose(rms, 1. / np.sqrt(2))

    def test_phaseshift(self):
        fs = 48000
        duration = 100e-3

        sig = Signal()
        sig.init_signal(2, duration, fs)
        sig.add_tone(100)
        sig[0].phase_shift(np.pi)

        test1 = audio.generate_tone(100, duration, fs, np.pi)
        test2 = audio.generate_tone(100, duration, fs)
        test = np.column_stack([test1, test2])

        testing.assert_almost_equal(sig.waveform, test)

    def test_cos_amp_modulator(self):
        fs = 48000
        sig = Signal().init_signal(1, 1, fs).add_tone(100)
        sig.add_cos_modulator(5, 1)

        test = audio.generate_tone(100, 1, fs)
        test *= audio.cos_amp_modulator(test, 5, fs)

        testing.assert_array_equal(sig.waveform, test)

        fs = 48000
        sig = Signal().init_signal(2, 1, fs).add_tone(100)
        sig.add_cos_modulator(5, 1)

        test = audio.generate_tone(100, 1, fs)
        test *= audio.cos_amp_modulator(test, 5, fs)

        testing.assert_array_equal(sig.waveform[:, 0], test)
        testing.assert_array_equal(sig.waveform[:, 1], test)

    def test_add_noise(self):
        fs = 48000
        sig = Signal().init_signal(1, 1, 48000).add_noise()

    def test_amplitude_spectrum(self):
        fs = 48000
        sig = Signal().init_signal(1, 1, 48000).add_tone(1e3)

        df = fs / sig.n_samples
        n_1000 = (sig.n_samples // 2) + int(1000 / df)

        a, b = sig.amplitude_spectrum()

        assert a[n_1000] == 1000
        testing.assert_almost_equal(np.abs(b[n_1000]), 0.5)

    def test_crest_factor(self):
        sig = Signal().init_signal(1, 1, 48000).add_tone(1e3)
        cfac = sig.calc_crest_factor()

        testing.assert_almost_equal(cfac, np.sqrt(2))
