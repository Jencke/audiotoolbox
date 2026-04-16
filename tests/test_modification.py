import pytest
import numpy as np
from numpy import testing
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


def test_set_dbspl_invertable():
    fs = 100e3
    signal = Signal(1, 1, fs).add_tone(100)
    signal.set_dbspl(15)
    testing.assert_almost_equal(signal.stats.dbspl, 15)

    signal.set_dbspl(0)
    testing.assert_almost_equal(signal.stats.rms, 20e-6)


def test_zeropad_both_args_raises():
    """Providing both number and duration must raise ValueError."""
    sig = Signal(1, 0.1, 48000)
    with pytest.raises(ValueError):
        sig.zeropad(number=1, duration=1e-3)


def test_set_dbfs_per_channel_behavior():
    """set_dbfs normalises independently per channel, which changes relative RMS.

    This is the current documented behavior. If per-channel=False global normalisation
    is added in the future, update this test accordingly.
    """
    rng = np.random.default_rng(42)
    sig = Signal(2, 1, 48000)
    sig[:, 0] = rng.standard_normal(sig.n_samples)
    sig[:, 1] = rng.standard_normal(sig.n_samples) * 4.0

    rms0_before = sig[:, 0].std()
    rms1_before = sig[:, 1].std()
    ratio_before = rms1_before / rms0_before

    sig.set_dbfs(-20)

    rms0_after = sig[:, 0].std()
    rms1_after = sig[:, 1].std()
    ratio_after = rms1_after / rms0_after

    # Each channel is normalised to -20 dBFS independently, so ratio collapses to ~1
    assert abs(rms0_after - rms1_after) < 1e-4, "Both channels should have equal RMS after per-channel normalisation"
    assert abs(ratio_before - ratio_after) > 0.1, "Ratio must change because normalisation is per-channel"
