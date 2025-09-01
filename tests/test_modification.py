import pytest
import numpy as np
from numpy import testing
from audiotoolbox import Signal

import numpy as np
import pytest
from audiotoolbox import Signal


def test_fade_window_hann_shape():
    """Test that fade window with 'hann' shape fades in and out correctly."""
    fs = 1000
    duration = 1.0
    rise_time = 0.2
    sig = Signal(1, duration, fs)
    sig[:] = 1.0
    sig.add_fade_window(rise_time, win_type="hann")
    n_samples = sig.n_samples
    n_fade = int(rise_time * fs)
    # Check fade-in and fade-out regions
    assert np.all(sig[:n_fade] < 1.0)
    assert np.all(sig[-n_fade:] < 1.0)
    # Check middle region remains unchanged
    assert np.allclose(sig[n_fade:-n_fade], 1.0)


def test_fade_window_cos_equivalence():
    """Test that 'cos' is equivalent to 'hann'."""
    fs = 1000
    duration = 1.0
    rise_time = 0.2
    sig1 = Signal(1, duration, fs)
    sig2 = Signal(1, duration, fs)
    sig1[:] = 1.0
    sig2[:] = 1.0
    sig1.add_fade_window(rise_time, win_type="cos")
    sig2.add_fade_window(rise_time, win_type="hann")
    assert np.allclose(sig1, sig2)


@pytest.mark.parametrize("channels", [1, (2,), (2, 2)])
def test_fade_window_multichannel(channels):
    """Test fade window works for multi-channel signals."""
    fs = 1000
    duration = 1.0
    rise_time = 0.2
    sig = Signal(channels, duration, fs)
    sig[:] = 1.0
    sig.add_fade_window(rise_time, win_type="hann")
    n_fade = int(rise_time * fs)
    # Check fade-in and fade-out regions for all channels
    assert np.all(sig[:n_fade] < 1.0)
    assert np.all(sig[-n_fade:] < 1.0)
    assert np.allclose(sig[n_fade:-n_fade], 1.0)


def test_fade_window_full_fade():
    """Test fade window with rise_time covering half the signal."""
    fs = 1000
    duration = 1.0
    rise_time = 0.5
    sig = Signal(1, duration, fs)
    sig[:] = 1.0
    sig.add_fade_window(rise_time, win_type="hann")
    # The middle should be all ones, edges faded
    n_fade = int(rise_time * fs)
    assert np.all(sig[:n_fade] < 1.0)
    assert np.all(sig[-n_fade:] < 1.0)
    assert np.allclose(sig[n_fade:-n_fade], 1.0)


def test_fade_window_invalid_type():
    """Test that an invalid window type raises an error in scipy."""
    fs = 1000
    duration = 1.0
    rise_time = 0.2
    sig = Signal(1, duration, fs)
    sig[:] = 1.0
    with pytest.raises(ValueError):
        sig.add_fade_window(rise_time, win_type="not_a_window")


def test_cos_amp_modulator_is_cos():
    mod = Signal(1, 1, 100e3)
    mod[:] = 1
    mod.add_cos_modulator(5, 1)
    test = mod.copy_empty().add_tone(5)

    testing.assert_array_almost_equal(mod, test + 1)
    assert max(mod) == 2.0


def test_cos_amp_modulator_is_mod_depth():
    mod = Signal(1, 1, 100e3)
    mod[:] = 1
    mod.add_cos_modulator(5, 0.5)
    assert mod[0] == 1.5


def test_cos_amp_modulator_start_phase():
    mod = Signal(1, 1, 100e3)
    mod[:] = 1
    mod.add_cos_modulator(5, 1, start_phase=np.pi / 4)
    test = mod.copy_empty().add_tone(5, start_phase=np.pi / 4)

    testing.assert_array_almost_equal(mod, test + 1)
    assert max(mod) == 2.0


def test_set_dbfs_reversible():
    signal = Signal(1, 1, 48000).add_tone(1000)
    # signal = audio.generate_tone(1000, 1, 48000)
    signal.set_dbfs(-5)
    testing.assert_almost_equal(signal.stats.dbfs, -5)


def test_set_dbfs_multichannel():
    signal = Signal((2, 3), 1, 48000).add_tone(1000)
    signal.ch[:, 2] *= 4
    signal.set_dbfs(-5)
    testing.assert_almost_equal(signal.stats.dbfs, -5)


def test_set_dbfs_peak():
    signal = Signal(1, 1, 48000).add_noise()
    signal.set_dbpeak(0)
    assert signal.abs().max() == 1.0

    signal.set_dbpeak(-6)
    assert signal.abs().max() == 10 ** (-6 / 20)


# signal = Signal(1, 1, 48000).add_tone(1000)
#     # signal = audio.generate_tone(1000, 1, 48000)
#     signal.set_dbfs(-5)
#     testing.assert_almost_equal(signal.stats.dbfs, -5)
# # RMS value of a -5 db sine
# m = (10 ** (-5 / 20)) / np.sqrt(2)

# signal = np.concatenate([-np.ones(10), np.ones(10)])
# signal = np.tile(signal, 100)
# signal = audio.set_dbfs(signal, -5)
# assert signal.max() == m

# assert audio.set_dbfs(2, 0, norm="peak") == 1
# signal = audio.generate_tone(1000, 8, 48000)
# assert audio.set_dbfs(signal, 0, "peak").max() == 1
# assert audio.set_dbfs(signal, -3, "peak").max() == 10 ** (-3 / 20)
