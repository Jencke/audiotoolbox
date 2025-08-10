import pytest
import numpy as np
from audiotoolbox import Signal


# Fixture to create a standard test signal (1-second, 1kHz sine wave at 48kHz)
@pytest.fixture
def sine_wave_signal():
    """Creates a 1-second, 1kHz sine wave signal at 48kHz fs."""
    fs = 48000
    duration = 1.0
    frequency = 1000
    sig = Signal(n_channels=1, duration=duration, fs=fs)
    sig.add_tone(frequency=frequency)
    return sig


# Fixture for a stereo signal
@pytest.fixture
def stereo_sine_wave_signal():
    """Creates a 1-second, 1kHz stereo sine wave signal at 48kHz fs."""
    fs = 48000
    duration = 1.0
    frequency = 1000
    sig = Signal(n_channels=2, duration=duration, fs=fs)
    sig.add_tone(frequency=frequency)
    return sig


def test_resample_downsampling(sine_wave_signal):
    """Test downsampling from a higher to a lower sampling rate."""
    original_fs = sine_wave_signal.fs
    new_fs = 24000
    original_samples = sine_wave_signal.n_samples

    # Resample the signal
    resampled_sig = sine_wave_signal.resample(new_fs)

    # Check that the method returns the instance for chaining
    assert resampled_sig is sine_wave_signal

    # Check that the fs attribute is updated
    assert resampled_sig.fs == new_fs

    # Check that the number of samples has changed as expected
    expected_samples = int(original_samples * (new_fs / original_fs))
    assert resampled_sig.n_samples == expected_samples
    assert resampled_sig.shape[0] == expected_samples

    # Verify the frequency content is preserved
    freqs = np.fft.rfftfreq(resampled_sig.n_samples, 1 / resampled_sig.fs)
    fft_vals = np.abs(np.fft.rfft(resampled_sig, axis=0))
    peak_freq_index = np.argmax(fft_vals)
    assert freqs[peak_freq_index] == pytest.approx(1000, abs=1)


def test_resample_upsampling(sine_wave_signal):
    """Test upsampling from a lower to a higher sampling rate."""
    original_fs = sine_wave_signal.fs
    new_fs = 96000
    original_samples = sine_wave_signal.n_samples

    # Resample the signal
    resampled_sig = sine_wave_signal.resample(new_fs)

    # Check that the fs attribute is updated
    assert resampled_sig.fs == new_fs

    # Check that the number of samples has changed as expected
    expected_samples = int(original_samples * (new_fs / original_fs))
    assert resampled_sig.n_samples == expected_samples
    assert resampled_sig.shape[0] == expected_samples


def test_resample_multichannel(stereo_sine_wave_signal):
    """Test that resampling works correctly for multi-channel signals."""
    new_fs = 22050
    resampled_sig = stereo_sine_wave_signal.resample(new_fs)

    assert resampled_sig.fs == new_fs
    assert resampled_sig.n_channels == 2
    assert resampled_sig.shape[1] == 2


def test_resample_raises_error_on_slice(sine_wave_signal):
    """Test that resample raises a RuntimeError when called on a slice."""
    # Create a view (slice) of the signal
    signal_view = sine_wave_signal.ch[0]

    with pytest.raises(RuntimeError, match="can only be applied to the whole signal"):
        signal_view.resample(24000)
