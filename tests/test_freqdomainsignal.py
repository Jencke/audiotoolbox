import unittest
#
from audiotools.oaudio import *
import audiotools as audio
import numpy as np
import numpy.testing as testing
import pytest

# import matplotlib.pyplot as plt


class test_oaudio(unittest.TestCase):

    def test_basics(self):
        sig = Signal(1, 1, 100)
        assert sig.n_channels == 1
        sig = sig.to_freqdomain()
        assert sig.n_channels == 1
        assert sig.freq.min() == -50
        assert sig.freq.max() == 49
        assert sig.omega.max() == 49 * np.pi * 2

        sig = FrequencyDomainSignal(1, 1, 100)
        assert sig.dtype == np.dtype('complex128')
        assert sig.n_channels == 1
        assert sig.freq.min() == -50
        assert sig.freq.max() == 49
        assert sig.omega.max() == 49 * np.pi * 2

    def test_slice(self):
        sig = FrequencyDomainSignal(2, 1, 100)
        assert sig._fs == 100
        sig2 = sig[:, 0]
        assert sig2._fs == 100

    def test_phaseshift(self):
        #phase shifting a full period
        sig = audio.Signal(2, 1, 48000).add_tone(100)
        sig[:, 1] = sig[:, 1].to_freqdomain().phase_shift(2*np.pi).to_timedomain()
        testing.assert_almost_equal(sig[:, 1], sig[:, 0])


        # phase shifting half a period
        sig = audio.Signal(2, 1, 48000)
        sig[:, 0].add_tone(100, start_phase=-0.5 * np.pi)
        sig[:, 1].add_tone(100)
        sig[:, 1] = sig[:, 1].to_freqdomain().phase_shift(0.5 * np.pi).to_timedomain()
        testing.assert_almost_equal(sig[:, 1], sig[:, 0])

    def test_timeshift(self):
        #timeshift a tone by one full phase
        sig = audio.Signal(2, 1, 48000).add_tone(100)
        sig[:, 1] = sig[:, 1].to_freqdomain().time_shift(1. / 100).to_timedomain()
        testing.assert_almost_equal(sig[:, 1], sig[:, 0])

        shift_samples = 500
        shift_time = shift_samples / 48000
        sig = audio.Signal(2, 1, 48000).add_noise()
        sig[:, 1] = sig[:, 1].to_freqdomain().time_shift(shift_time).to_timedomain()
        testing.assert_almost_equal(sig[shift_samples:, 1], sig[:-shift_samples, 0])

        shift_time = 3.2e-4
        sig = audio.Signal(2, 1.000, 48000).add_noise()
        res =  sig[:, 1].to_freqdomain().time_shift(shift_time).to_timedomain()
        assert np.all(np.isreal(res))

    def test_analytical(self):
        sig = audio.Signal(2, 1, 48000).add_tone(100).to_freqdomain()
        asig = sig.to_analytical()
        testing.assert_almost_equal(sig.to_timedomain(), asig.to_timedomain())

        sig = audio.Signal(2, 1, 48000).add_noise().to_freqdomain()
        sig2 = sig.to_analytical()
