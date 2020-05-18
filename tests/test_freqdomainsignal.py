import unittest

from audiotools.oaudio import *
import audiotools as audio
import numpy as np
import numpy.testing as testing
import pytest

# import matplotlib.pyplot as plt


class test_oaudio(unittest.TestCase):

    def test_set_waveform(self):
        sig = FrequencyDomainSignal()
        assert sig.n_channels == 0
        assert sig.n_samples == 0

        # Test set single channel signal
        sig.set_waveform(np.zeros(100), 100)
        assert sig.n_channels == 1
        assert sig.n_samples == 100
        assert len(sig.freq) == sig.n_samples
        assert sig.freq.max() == 49.0
        assert sig.freq.min() == -50.0
        assert sig.duration == 1

    def test_multiply(self):
        sig = Signal(2, 1, 48e3).add_noise().to_freqdomain()
        assert testing.assert_almost_equal(sig[0].waveform, sig[1].waveform)



        # # test multiple channel signal
        # sig = Signal()
        # sig.set_waveform(np.zeros([100, 2]), 100)
        # assert sig.n_channels == 2
        # assert sig.n_samples == 100
        # assert sig.duration == 1
