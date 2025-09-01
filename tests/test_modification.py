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
