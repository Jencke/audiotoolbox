import unittest

from audiotools.oaudio import *
import audiotools as audio
import numpy as np
import numpy.testing as testing
import pytest


class test_oaudio(unittest.TestCase):
    def test_init_signal(self):
        sig = Signal(1, 100, 1)

        assert sig.fs == 1
        assert sig.duration == 100
        assert sig.n_samples == 100

        sig = Signal(1, 100, 1)
        assert sig.fs == 1
        assert sig.duration == 100
        assert sig.n_samples == 100

    def test_multidim(self):
        sig = Signal((200, 2), 100, 1)

        assert sig.n_samples == 100
        assert sig.n_channels == (200, 2)

    def test_time(self):
        sig = Signal(1, 100, 1)
        time = sig.time

        assert sig.time[0] == 0
        assert np.all(np.diff(time) == 1)
        assert sig.time[-1] == 99

    def test_addtone(self):
        fs = 48000
        duration = 100e-3

        sig = Signal(1, duration, fs)
        sig.add_tone(100)
        sig.add_tone(200, start_phase=np.pi)

        test = audio.generate_tone(100, duration, fs)
        test += audio.generate_tone(200, duration, fs, np.pi)

        testing.assert_equal(sig, test)

        sig = Signal(1, duration, fs)
        sig.add_tone(100, amplitude=2)

        test = 2 * audio.generate_tone(100, duration, fs)
        testing.assert_equal(sig, test)

        sig = Signal(2, duration, fs)
        sig.add_tone(100, amplitude=2)

        test = 2 * audio.generate_tone(100, duration, fs)
        testing.assert_equal(sig.ch[0], test)
        testing.assert_equal(sig.ch[1], test)

        sig = Signal((2, 2), duration, fs)
        sig.add_tone(100, amplitude=2)

        test = 2 * audio.generate_tone(100, duration, fs)
        testing.assert_equal(sig.ch[0, 0], test)
        testing.assert_equal(sig.ch[1, 0], test)
        testing.assert_equal(sig.ch[0, 1], test)
        testing.assert_equal(sig.ch[1, 1], test)


    def test_setdbspl(self):
        fs = 48000
        duration = 100e-3

        sig = Signal(1, duration, fs)
        sig.add_tone(100).set_dbspl(50)

        test = audio.generate_tone(100, duration, fs)
        test = audio.set_dbspl(test, 50)

        testing.assert_equal(sig, test)

    def test_calcdbspl(self):
        fs = 48000
        duration = 100e-3

        sig = Signal(1, duration, fs)
        sig.add_tone(100).set_dbspl(50)
        db = sig.calc_dbspl()

        assert db == 50

    def test_setdbfs_calcdbfs(self):
        fs = 48000
        duration = 100e-3

        sig = Signal(1, duration, fs)
        sig.add_tone(100).set_dbfs(-5)

        assert(audio.calc_dbfs(sig) == -5)
        assert(sig.calc_dbfs() == -5)


    def test_zeropad(self):
        fs = 48000
        duration = 100e-3

        sig = Signal(1, duration, fs)
        sig.add_tone(100).zeropad(number=10)
        assert np.all(sig[:10] == sig[-10:])
        assert np.all(sig[:10] == 0)

        sig = Signal(1, duration, fs)
        sig.add_tone(100).zeropad(number=[10, 5])
        assert np.all(sig[:10] == 0)
        assert np.all(sig[-5:] == 0)

        sig = Signal(1, duration, fs)
        sig.add_tone(100).zeropad(duration=10e-3)
        n_zeros = audio.nsamples(10e-3, fs)
        assert np.all(sig[:n_zeros] == 0)
        assert np.all(sig[-n_zeros:] == 0)

        sig = Signal(1, duration, fs)
        sig.add_tone(100).zeropad(duration=[5e-3, 10e-3])
        n_zeros_s = audio.nsamples(5e-3, fs)
        n_zeros_e = audio.nsamples(10e-3, fs)
        assert np.all(sig[:n_zeros_s] == 0)
        assert np.all(sig[-n_zeros_e:] == 0)

    def test_fadewindow(self):
        fs = 48000
        duration = 100e-3

        sig = Signal(1, duration, fs)
        sig.add_tone(100).add_fade_window(rise_time=10e-3, type='gauss')
        test = audio.generate_tone(100, duration, fs)
        test *= audio.gaussian_fade_window(test, 10e-3, fs)
        testing.assert_equal(sig, test)

        sig = Signal(1, duration, fs)
        sig.add_tone(100).add_fade_window(rise_time=10e-3, type='cos')
        test = audio.generate_tone(100, duration, fs)
        test *= audio.cosine_fade_window(test, 10e-3, fs)
        testing.assert_equal(sig, test)

        sig = Signal(1, duration, fs)
        sig.add_tone(100).add_fade_window(rise_time=10e-3, type='cos')
        test = audio.generate_tone(100, duration, fs)
        test *= audio.cosine_fade_window(test, 10e-3, fs)
        testing.assert_equal(sig, test)

    def test_add(self):
        fs = 48000
        duration = 100e-3

        # test addition of signal
        sig = Signal(1, duration, fs)
        sig.add_tone(100)
        sig2 = Signal(1, duration, fs)
        sig2.add_tone(200)
        sig = sig + sig2
        # sig.add(sig2)
        test = Signal(1, duration, fs)
        test.add_tone(100).add_tone(200)
        testing.assert_equal(sig, test)

        sig = Signal(1, duration, fs)
        sig.add_tone(100)
        sig += 2
        sig += 1
        test = Signal(1, duration, fs)
        test.add_tone(100)
        testing.assert_almost_equal(sig, test + 3.0)

        sig = Signal(2, duration, fs)
        sig.add_tone(100)
        sig += np.array([1, 2])
        testing.assert_allclose(sig[:,1].mean() - sig[:,0].mean(), 1)

        sig = Signal(2, duration, fs)
        sig.add_tone(100)
        sig[:, 1] += sig[:,0]
        testing.assert_allclose(sig[:,1] /  sig[:,0], 2)


    def test_multiply(self):
        fs = 48000
        duration = 100e-3

        # test addition of signal
        sig = Signal(1, duration, fs)
        sig.add_tone(100)
        sig2 = Signal(1, duration, fs)
        sig2.add_tone(100)
        sig *= sig2
        testing.assert_almost_equal(sig, sig2**2)

        sig = Signal(1, duration, fs)
        sig.add_tone(100)
        sig.multiply(2).multiply(2.1)
        test = Signal(1, duration, fs)
        test.add_tone(100)
        testing.assert_almost_equal(sig, test * 2 * 2.1)

        sig = Signal(2, duration, fs)
        sig.add_tone(100)
        sig.multiply(np.array([1, 2]))
        testing.assert_almost_equal(sig[:, 1], sig[:, 0] * 2)

        sig = Signal(2, duration ,fs)
        sig.add_tone(100)
        sig[:, 1].multiply(sig[:, 0])
        testing.assert_almost_equal(sig[:, 1],  sig[:, 0]**2)

    def test_mean(self):
        sig = Signal(2, 100e-3, 100e3)
        sig.add_tone(100)
        sig += np.array([1, 2])
        mean = sig.mean(0)
        testing.assert_almost_equal(mean, np.array([1, 2]))

    def test_rms(self):
        sig = Signal(2, 100e-3, 100e3)
        sig.add_tone(100)
        rms = sig.rms()
        testing.assert_allclose(rms, 1. / np.sqrt(2))

    def test_delay(self):
        fs = 48000
        duration = 100e-3

        # test sample shift function
        shift_samples = 500
        shift_time = shift_samples / 48000
        sig = Signal(2, 1, 48000).add_noise()
        sig[:, 1].delay(shift_time, method='sample')
        testing.assert_almost_equal(sig[:-shift_samples, 0], sig[shift_samples:, 1])

        # sample shift multiple dimensions
        shift_samples = 500
        shift_time = shift_samples / 48000
        sig = Signal((2, 2), 1, 48000).add_noise()
        sig[:, 1].delay(shift_time, method='sample')
        testing.assert_almost_equal(sig[:-shift_samples, 0], sig[shift_samples:, 1])

        shift_samples = 500
        shift_time = shift_samples / 48000
        sig = Signal(2, 1, 48000).add_noise()
        sig[:, 1].delay(shift_time, method='fft')
        testing.assert_almost_equal(sig[:-shift_samples, 0], sig[shift_samples:, 1])

        shift_samples = 500
        shift_time = shift_samples / 48000
        sig = Signal(2, 1, 48000).add_noise()
        sig[:, 1].delay(shift_time, method='fft')
        sig[:, 0].delay(shift_time, method='sample')
        testing.assert_almost_equal(sig[:, 0], sig[:, 1])



    def test_phaseshift(self):
        fs = 48000
        duration = 100e-3

        sig = Signal(2, duration, fs)
        sig.add_tone(100)
        sig[:, 0].phase_shift(np.pi)

        test1 = audio.generate_tone(100, duration, fs, np.pi)
        test2 = audio.generate_tone(100, duration, fs)
        test = np.column_stack([test1, test2])

        testing.assert_almost_equal(sig, test)

    def test_cos_amp_modulator(self):
        fs = 48000
        sig = Signal(1, 1, fs).add_tone(100)
        sig.add_cos_modulator(5, 1)

        test = audio.generate_tone(100, 1, fs)
        test *= audio.cos_amp_modulator(test, 5, fs)

        testing.assert_array_equal(sig, test)

        fs = 48000
        sig = Signal(2, 1, fs).add_tone(100)
        sig.add_cos_modulator(5, 1)

        test = audio.generate_tone(100, 1, fs)
        test *= audio.cos_amp_modulator(test, 5, fs)

        testing.assert_array_equal(sig[:, 0], test)
        testing.assert_array_equal(sig[:, 1], test)

    def test_add_noise(self):
        fs = 48000
        sig = Signal(1, 1, 48000).add_noise()
        assert sig.max() != 0
        sig = Signal(2, 1, 48000).add_noise()
        assert np.all(sig.max(axis=0) != 0)

        sig = Signal((2, 2), 1, 48000).add_noise()
        assert np.all(sig.max(axis=0) != 0)

        sig = Signal((2, 2), 1, 48000).add_noise(variance = 2)
        assert np.var(sig.ch[0]) == np.var(sig.ch[1])
        testing.assert_almost_equal(np.var(sig), 2)

    def test_add_noise(self):
        fs = 48000
        sig = Signal(1, 1, 48000).add_noise()
        assert sig.max() != 0
        sig = Signal(2, 1, 48000).add_noise()
        assert np.all(sig.max(axis=0) != 0)

        sig = Signal((2, 2), 1, 48000).add_noise()
        assert np.all(sig.max(axis=0) != 0)

        sig = Signal((2, 2), 1, 48000).add_noise(variance = 2)
        assert np.var(sig.ch[0]) == np.var(sig.ch[1])
        testing.assert_almost_equal(np.var(sig), 2)


    def test_add_uncorr_noise(self):
        fs = 48000
        sig = Signal(5, 1, fs).add_uncorr_noise()
        # lower trianglular matrix should be  0
        testing.assert_almost_equal(np.tril(np.cov(sig.T), -1), 0)

        #Multidimensional case
        sig = Signal((2, 2), 1, fs).add_uncorr_noise()
        assert(sig.n_channels == (2, 2))


    def test_crest_factor(self):
        sig = Signal(1, 1, 48000).add_tone(1e3)
        cfac = sig.calc_crest_factor()

        testing.assert_almost_equal(cfac, np.sqrt(2))

    def test_clip(self):
        sig = Signal(2, 1, 48000).add_noise()
        o_sig = sig.copy()
        sig.clip(0, 1)
        assert sig.n_samples == o_sig.n_samples

        # Test multi channel clipping (2 x 2 )
        sig = Signal((2, 2), 1, 48000).add_noise()
        o_sig = sig.copy()
        sig.clip(0, 1)
        assert sig.n_samples == o_sig.n_samples

        sig = Signal(2, 1, 48000).add_noise()
        o_sig = sig.copy()
        sig.clip(0, 0.5)
        assert sig.n_samples == (o_sig.n_samples // 2)
        assert np.all(sig == o_sig[:o_sig.n_samples // 2, :])
        assert sig.base == None

        sig = Signal(2, 1, 48000).add_noise()
        o_sig = sig.copy()
        sig.clip(0.5)
        assert sig.n_samples == (o_sig.n_samples // 2)
        assert np.all(sig == o_sig[o_sig.n_samples // 2:, :])
        assert sig.base == None

        sig = Signal((2, 2), 1, 48000).add_noise()
        o_sig = sig.copy()
        sig.clip(0.5)
        assert sig.n_samples == (o_sig.n_samples // 2)
        assert np.all(sig == o_sig[o_sig.n_samples // 2:, :])
        assert sig.base == None

        # test negative indexing
        sig = Signal(2, 1, 48000).add_noise()
        o_sig = sig.copy()
        n_samples = audio.nsamples(0.9, sig.fs)
        sig.clip(0, -0.1)
        assert sig.n_samples == n_samples
        assert np.all(sig == o_sig[:n_samples, :])
        assert sig.base == None

    def test_concatenate(self):
        sig_a = Signal(2, 1, 48000).add_noise()
        old_n = sig_a.n_samples
        sig_b = Signal(2, 0.5, 48000).add_noise()

        sig_a.concatenate(sig_b)
        assert sig_a.n_samples == old_n + sig_b.n_samples
        testing.assert_equal(sig_a[old_n:], sig_b)

    def test_bandpass_brickwall(self):
        sig = audio.Signal((2, 2), 1, 48000).add_noise().bandpass(500, 100, 'brickwall')
        sig = sig.to_freqdomain()
        testing.assert_array_almost_equal(sig[np.abs(sig.freq) > 550], 0)
        testing.assert_array_almost_equal(sig[np.abs(sig.freq) < 450], 0)
        assert np.all(sig[(np.abs(sig.freq) < 550) & (np.abs(sig.freq) > 450)] != 0)

    def test_bandpass_gammatone(self):
        # check real valued output
        sig = audio.Signal(1, 1, 48000).add_tone(500)
        sig2 = sig.copy()
        sig.bandpass(500, 100, 'gammatone')
        assert not np.iscomplexobj(sig)
        assert sig.shape == sig2.shape

        # check complex output
        sig = audio.Signal(1, 1, 48000).add_tone(500)
        sig2 = sig.copy()
        sig.bandpass(500, 100, 'gammatone', return_complex=True)
        assert np.iscomplexobj(sig)
        assert sig.shape == sig2.shape

        # check equivalence of real and complex results
        sig = audio.Signal(1, 1, 48000).add_tone(500)
        sig2 = sig.copy()
        sig.bandpass(500, 100, 'gammatone')
        sig2.bandpass(500, 100, 'gammatone', return_complex=True)
        testing.assert_array_equal(sig2.real, sig)

        # check kwargs
        sig = audio.Signal(1, 1, 48000).add_tone(500)
        out = audio.filter.gammatone(sig, sig2.fs, 500, 100, order=2, attenuation_db=-1)
        sig.bandpass(500, 100, 'gammatone', order=2, attenuation_db=-1)
        testing.assert_array_equal(sig, out.real)

    def test_channel_indexing(self):
        sig = Signal((2, 2), 1, 48000).add_noise()
        testing.assert_equal(sig.ch[0, 0], sig[:, 0, 0])
        testing.assert_equal(sig.ch[0], sig[:, 0])

        sig = Signal(2, 1, 48000)
        sig.ch[0] = 1
        assert np.all(sig[:, 0] == 1)

        sig.ch[1].add_tone(500)
        tone_2 = audio.generate_tone(500, sig.duration, sig.fs)
        testing.assert_equal(sig.ch[1], tone_2)

    def test_analytical(self):
        sig = audio.Signal((2, 2), 1, 48000).add_noise()
        asig = sig.to_analytical()
        testing.assert_almost_equal(sig, asig.real)
