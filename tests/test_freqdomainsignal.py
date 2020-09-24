import unittest

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
        assert sig.n_channels == 1
        assert sig.freq.min() == -50
        assert sig.freq.max() == 49
        assert sig.omega.max() == 49 * np.pi * 2

    def test_slice(self):
        sig = FrequencyDomainSignal(2, 1, 100)
        assert sig._fs == 100
        sig2 = sig[:, 0]
        assert sig2._fs == 100



    #     # Test set single channel signal
    #     sig.set_waveform(np.zeros(100), 100)
    #     assert sig.n_channels == 1
    #     assert sig.n_samples == 100
    #     assert len(sig.freq) == sig.n_samples
    #     assert sig.freq.max() == 49.0
    #     assert sig.freq.min() == -50.0
    #     assert sig.duration == 1

    # def test_phase_shift(self):
    #     sig = Signal(1, 1, 48000).to_freqdomain()
    #     assert np.all(sig.phase == 0)

    #     sig.phase_shift(0.2 * np.pi)
    #     assert np.all(sig.phase == (0.2 * np.pi))

        # sig = sig.to_timedomain().add_noise().to_freqdomain()
        # assert ~np.all(sig.phase == 0)

        # orig_phases = sig.phase
        # shift = 0.2 * np.pi
        # sig.phase_shift(shift)
        # print((sig.phase - orig_phases) == shift)

    # def test_init_signal(self):
    #     sig = FrequencyDomainSignal()
    #     sig.init_signal(1, 100, 1)

    #     assert sig.fs == 1
    #     assert sig.duration == 100
    #     assert sig.n_samples == 100

    #     sig = FrequencyDomainSignal(1, 100, 1)
    #     assert sig.fs == 1
    #     assert sig.duration == 100
    #     assert sig.n_samples == 100

    pass
