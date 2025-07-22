import pytest
import numpy as np
from audiotoolbox.oaudio.signal import Signal


# Fixture for a simple mono signal
@pytest.fixture
def mono_signal():
    """Create a mono signal for testing."""
    fs = 48000
    duration = 1.0
    n_samples = int(duration * fs)
    sig = Signal(n_channels=1, duration=duration, fs=fs)
    sig[:] = np.arange(n_samples)
    return sig


@pytest.fixture
def stereo_signal():
    """Create a stereo signal for testing."""
    fs = 48000
    duration = 1.0
    n_samples = int(duration * fs)
    sig = Signal(n_channels=2, duration=duration, fs=fs)
    # Use different ranges for each channel to distinguish them
    ch1 = np.arange(n_samples)
    ch2 = np.arange(n_samples, 2 * n_samples)
    sig[:] = np.vstack([ch1, ch2]).T
    return sig


def test_as_blocked_mono_perfect_fit(mono_signal):
    """
    Test as_blocked on a mono signal that fits perfectly without padding.
    """
    # Signal length is 48000. (48000 - 1024) / 512 + 1 = 92.75 -> 93 blocks
    # Required length = ceil((48000 - 1024) / 512) * 512 + 1024 = 92 * 512 + 1024 = 48128
    # The logic in the code seems to pad even when it's not strictly necessary
    # Let's use a length that works perfectly
    sig = mono_signal[:1536].copy()  # length = 1536
    block_size = 1024
    overlap = 512
    step = block_size - overlap

    # Expected number of windows: (1536 - 1024) / 512 + 1 = 2
    blocked_sig = sig.as_blocked(block_size=block_size, overlap=overlap)

    assert isinstance(blocked_sig, Signal)
    assert blocked_sig.shape == (block_size, 2)
    assert blocked_sig.fs == sig.fs

    # Check content of the first block
    np.testing.assert_array_equal(blocked_sig[:, 0], sig[0:block_size])
    # Check content of the second block
    np.testing.assert_array_equal(blocked_sig[:, 1], sig[step : step + block_size])


def test_as_blocked_stereo_perfect_fit(stereo_signal):
    """
    Test as_blocked on a stereo signal that fits perfectly.
    """
    sig = stereo_signal[:1536].copy()  # length = 1536
    block_size = 1024
    overlap = 512
    step = block_size - overlap

    blocked_sig = sig.as_blocked(block_size=block_size, overlap=overlap)

    assert isinstance(blocked_sig, Signal)
    assert blocked_sig.shape == (block_size, 2, 2)
    assert blocked_sig.fs == sig.fs

    # Check content of the first block, first channel
    np.testing.assert_array_equal(blocked_sig[:, 0, 0], sig[0:block_size, 0])
    # Check content of the second block, second channel
    np.testing.assert_array_equal(
        blocked_sig[:, 1, 1], sig[step : step + block_size, 1]
    )


def test_as_blocked_mono_with_padding(mono_signal):
    """
    Test as_blocked on a mono signal that requires padding.
    """
    sig = mono_signal[:2000].copy()  # Use a length that doesn't fit perfectly
    original_length = sig.n_samples
    block_size = 1024
    overlap = 512
    step = block_size - overlap

    # Required length = ceil((2000 - 1024) / 512) * 512 + 1024 = 2 * 512 + 1024 = 2048
    # n_pad = 2048 - 2000 = 48
    with pytest.warns(UserWarning, match="Zero padding 48 samples"):
        blocked_sig = sig.as_blocked(block_size=block_size, overlap=overlap)

    # The original signal is padded in-place
    assert sig.n_samples == 2048
    assert blocked_sig.shape == (block_size, 3)

    # Check that the padded part is zero
    np.testing.assert_array_equal(sig[original_length:], np.zeros(48))

    # Check the last block's content
    last_block_start = 2 * step
    np.testing.assert_array_equal(
        blocked_sig[:, -1], sig[last_block_start : last_block_start + block_size]
    )


def test_as_blocked_no_overlap(mono_signal):
    """
    Test as_blocked with zero overlap.
    """
    sig = mono_signal[:4096].copy()  # 4 blocks of 1024
    block_size = 1024
    overlap = 0
    step = block_size - overlap

    blocked_sig = sig.as_blocked(block_size=block_size, overlap=overlap)

    assert blocked_sig.shape == (block_size, 4)

    # Check the third block
    np.testing.assert_array_equal(blocked_sig[:, 2], sig[2 * step : 3 * step])
