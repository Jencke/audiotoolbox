import audiotools.wav as wav
import audiotools as audio
import numpy as np
import numpy.testing as testing

import pytest


def test_writewav_readwav():
    """Test invertability of readwav and writewav"""
    fs = 41200
    signal = audio.Signal(2, 2, fs)
    signal[:] = np.linspace(-1, 1, signal.n_samples)[:, None]

    bitdepth = [8, 16]
    for bd in bitdepth:
        wav.writewav("test.wav", signal, signal.fs, bd)
        out8, fs = wav.readwav("test.wav")
        testing.assert_allclose(out8, signal, atol=2 / 2**bd, rtol=1)


def test_writewav():
    """Test invertability of readwav and writewav"""
    fs = 41200
    signal = audio.Signal(2, 2, fs)
    signal[:] = np.linspace(-1, 1, signal.n_samples)[:, None]

    bitdepth = [8, 16, 32]
    for bd in bitdepth:
        wav.writewav("test.wav", signal, signal.fs, bd)

    with pytest.raises(ValueError):
        wav.writewav(filename="test.wav", signal=signal, fs=signal.fs, bitdepth=7)

    with pytest.raises(ValueError):
        wav.writewav(filename="test.wav", signal=signal, fs=signal.fs, bitdepth=64)


def test_int_to_fullscale():
    """correct conversion from integer to fullscale"""
    bitdepth = [8, 16]
    for bd in bitdepth:
        i_max = 2**bd // 2
        array = np.arange(-i_max + 1, i_max, 1)
        sig_out = np.linspace(-1, 1, len(array))
        out = wav.int_to_fullscale(array, bd)
        testing.assert_almost_equal(sig_out, out)
