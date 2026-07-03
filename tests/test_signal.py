from audiotoolbox import Signal
import audiotoolbox as audio
import numpy as np
import numpy.testing as testing
import pytest
import warnings


def _channel_indices(signal):
    channel_shape = signal.shape[1:]
    if not channel_shape:
        return [()]
    return list(np.ndindex(channel_shape))


def _assert_all_channels_equal(signal, expected):
    for idx in _channel_indices(signal):
        channel = signal if idx == () else signal.ch[idx]
        expected_arr = np.asarray(expected)
        if channel.ndim == 2 and channel.shape[1] == 1 and expected_arr.ndim == 1:
            expected_arr = expected_arr[:, np.newaxis]
        testing.assert_almost_equal(channel, expected_arr)


def _assert_vectorized_addtone_matches_iterative(frequencies, amplitudes, start_phases):
    duration = 100e-3
    fs = 48000
    target_length = max(
        np.size(frequencies), np.size(amplitudes), np.size(start_phases)
    )
    freqs = np.broadcast_to(np.asarray(frequencies), (target_length,))
    amps = np.broadcast_to(np.asarray(amplitudes), (target_length,))
    phases = np.broadcast_to(np.asarray(start_phases), (target_length,))

    sig = audio.Signal((2, 3), duration, fs)
    for freq, amplitude, start_phase in zip(freqs, amps, phases):
        sig.add_tone(frequency=freq, amplitude=amplitude, start_phase=start_phase)

    sig2 = audio.Signal((2, 3), duration, fs)
    sig2.add_tone(frequencies, amplitudes, start_phases)
    testing.assert_almost_equal(sig, sig2)


def _assert_convolution_shape(
    signal_channels, kernel_channels, expected_channels, overlap_dimensions=True
):
    sig = audio.Signal(signal_channels, 1, 48000).add_noise()
    kernel = audio.Signal(kernel_channels, 100e-3, 48000)
    sig.convolve(kernel, overlap_dimensions=overlap_dimensions)
    assert sig.n_channels == expected_channels


def test_init_signal():
    sig = Signal(1, 100, 1)

    assert sig.fs == 1
    assert sig.duration == 100
    assert sig.n_samples == 100

    sig = Signal(1, 100, 1)
    assert sig.fs == 1
    assert sig.duration == 100
    assert sig.n_samples == 100


def test_multidim():
    sig = Signal((200, 2), 100, 1)

    assert sig.n_samples == 100
    assert sig.n_channels == (200, 2)


def test_time():
    sig = Signal(1, 100, 1)
    time = sig.time

    assert sig.time[0] == 0
    assert np.all(np.diff(time) == 1)
    assert sig.time[-1] == 99


def test_addtone_superposition():
    fs = 48000
    duration = 100e-3

    sig = Signal(1, duration, fs)
    sig.add_tone(100)
    sig.add_tone(200, start_phase=np.pi)

    test = np.cos(2 * np.pi * sig.time * 100)
    test += np.cos(2 * np.pi * sig.time * 200 + np.pi)

    testing.assert_almost_equal(sig[:, 0], test)


def test_addtone_amplitude():
    fs = 48000
    duration = 100e-3

    sig = Signal(1, duration, fs)
    sig.add_tone(100, amplitude=2)

    test = 2 * np.cos(2 * np.pi * sig.time * 100)
    testing.assert_almost_equal(sig[:, 0], test)


@pytest.mark.parametrize("channels", [2, (2, 2)])
def test_addtone_applies_to_all_channels(channels):
    fs = 48000
    duration = 100e-3

    sig = Signal(channels, duration, fs)
    sig.add_tone(100, amplitude=2)
    test = 2 * np.cos(2 * np.pi * sig.time * 100)
    _assert_all_channels_equal(sig, test)


@pytest.mark.parametrize(
    ("frequencies", "amplitudes", "start_phases"),
    [
        ([150.0, 300.0, 450.0], [0.3, 0.6, 0.9], [0.0, np.pi / 4, np.pi / 2]),
        (np.array([220.0]), [0.2, 0.4, 0.6], [0.1, 0.2, 0.3]),
        ([120.0, 240.0, 360.0], np.array([0.75]), [0.1, 0.2, 0.3]),
        ([180.0, 360.0, 540.0], [0.5, 1.0, 1.5], np.array([np.pi / 3])),
    ],
)
def test_addtone_vectorized_matches_iterative(frequencies, amplitudes, start_phases):
    _assert_vectorized_addtone_matches_iterative(frequencies, amplitudes, start_phases)


def test_stats():
    sig = Signal(1, 1, 48000)
    assert hasattr(sig, "stats")

    sig = Signal(1, 1, 48000)
    sig = sig.copy()
    assert hasattr(sig, "stats")


def test_zeropad():
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


def test_zeropad_raises_when_number_and_duration_are_both_given():
    sig = Signal(1, 100e-3, 48000)

    with pytest.raises(ValueError, match="Must state only duration or number of zeros"):
        sig.zeropad(number=1, duration=1e-3)


def test_add():
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
    testing.assert_allclose(sig[:, 1].mean() - sig[:, 0].mean(), 1)

    sig = Signal(2, duration, fs)
    sig.add_tone(100)
    sig[:, 1] += sig[:, 0]
    testing.assert_allclose(sig[:, 1] / sig[:, 0], 2)


def test_multiply():
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

    sig = Signal(2, duration, fs)
    sig.add_tone(100)
    sig[:, 1].multiply(sig[:, 0])
    testing.assert_almost_equal(sig[:, 1], sig[:, 0] ** 2)


def test_mean():
    sig = Signal(2, 100e-3, 100e3)
    sig.add_tone(100)
    sig += np.array([1, 2])
    mean = sig.mean(0)
    testing.assert_almost_equal(mean, np.array([1, 2]))


def test_delay():
    fs = 48000
    duration = 100e-3

    # test sample shift function
    shift_samples = 500
    shift_time = shift_samples / 48000
    sig = Signal(2, 1, 48000).add_noise()
    sig[:, 1].delay(shift_time, method="sample")
    testing.assert_almost_equal(sig[:-shift_samples, 0], sig[shift_samples:, 1])

    # sample shift multiple dimensions
    shift_samples = 500
    shift_time = shift_samples / 48000
    sig = Signal((2, 2), 1, 48000).add_noise()
    sig[:, 1].delay(shift_time, method="sample")
    testing.assert_almost_equal(sig[:-shift_samples, 0], sig[shift_samples:, 1])

    shift_samples = 500
    shift_time = shift_samples / 48000
    sig = Signal(2, 1, 48000).add_noise()
    sig[:, 1].delay(shift_time, method="fft")
    testing.assert_almost_equal(sig[:-shift_samples, 0], sig[shift_samples:, 1])

    shift_samples = 500
    shift_time = shift_samples / 48000
    sig = Signal(2, 1, 48000).add_noise()
    sig[:, 1].delay(shift_time, method="fft")
    sig[:, 0].delay(shift_time, method="sample")
    testing.assert_almost_equal(sig[:, 0], sig[:, 1])


def test_phaseshift():
    fs = 48000
    duration = 100e-3

    sig = Signal(2, duration, fs)
    sig.add_tone(100)
    sig[:, 0].phase_shift(np.pi)

    test = audio.Signal(2, duration, fs)
    test.ch[0].add_tone(100, start_phase=np.pi)
    test.ch[1].add_tone(100)

    testing.assert_almost_equal(sig, test)


def test_trim():
    sig = Signal(2, 1, 48000).add_noise()
    o_sig = sig.copy()
    sig.trim(0, 1)
    assert sig.n_samples == o_sig.n_samples

    sig = Signal(1, 1, 48000).add_noise()
    o_sig = sig.copy()
    sig.trim(0, 0.5)
    assert sig.n_samples == o_sig.n_samples // 2

    # Test multi channel trimming (2 x 2 )
    sig = Signal((2, 2), 1, 48000).add_noise()
    o_sig = sig.copy()
    sig.trim(0, 1)
    assert sig.n_samples == o_sig.n_samples

    sig = Signal(2, 1, 48000).add_noise()
    o_sig = sig.copy()
    sig.trim(0, 0.5)
    assert sig.duration == 0.5
    assert sig.n_samples == (o_sig.n_samples // 2)
    assert np.all(sig == o_sig[: o_sig.n_samples // 2, :])
    assert sig.base == None

    sig = Signal(2, 1, 48000).add_noise()
    o_sig = sig.copy()
    sig.trim(0.5)
    assert sig.n_samples == (o_sig.n_samples // 2)
    assert np.all(sig == o_sig[o_sig.n_samples // 2 :, :])
    assert sig.base == None

    sig = Signal((2, 2), 1, 48000).add_noise()
    o_sig = sig.copy()
    sig.trim(0.5)
    assert sig.n_samples == (o_sig.n_samples // 2)
    assert np.all(sig == o_sig[o_sig.n_samples // 2 :, :])
    assert sig.base == None

    # test negative indexing
    sig = Signal(2, 1, 48000).add_noise()
    o_sig = sig.copy()
    n_samples = audio.nsamples(0.9, sig.fs)
    sig.trim(0, -0.1)
    assert sig.n_samples == n_samples
    assert np.all(sig == o_sig[:n_samples, :])
    assert sig.base == None


def test_remove_silence_mono_blockwise():
    fs = 1000
    sig = Signal(1, 300e-3, fs)
    sig[100:200] = 1.0

    sig.remove_silence(
        threshold_dbfs=-40,
        block_duration=50e-3,
        overlap_duration=10e-3,
    )

    # The three non-silent blocks span samples 80..209.
    assert sig.n_samples == 130
    assert np.sum(sig == 1.0) == 100


def test_remove_silence_multichannel_keeps_alignment():
    fs = 1000
    sig = Signal(2, 300e-3, fs)
    sig[100:200, 0] = 1.0

    sig.remove_silence(
        threshold_dbfs=-40,
        block_duration=50e-3,
        overlap_duration=10e-3,
    )

    assert sig.n_samples == 130
    assert np.sum(sig[:, 0] == 1.0) == 100
    assert np.all(sig[:, 1] == 0.0)


def test_remove_silence_edges_only_keeps_inner_silence():
    fs = 1000
    sig = Signal(1, 500e-3, fs)
    sig[50:120] = 1.0
    sig[300:370] = 1.0

    full_remove = sig.copy()
    full_remove.remove_silence(
        threshold_dbfs=-40,
        block_duration=50e-3,
        overlap_duration=10e-3,
    )

    edges_only = sig.copy()
    edges_only.remove_silence(
        threshold_dbfs=-40,
        block_duration=50e-3,
        overlap_duration=10e-3,
        edges_only=True,
    )

    # Block settings produce active spans 40..129 and 280..409.
    # Edges-only trimming keeps the full 40..409 range.
    assert full_remove.n_samples == 220
    assert edges_only.n_samples == 370
    assert edges_only.n_samples > full_remove.n_samples


def test_remove_silence_validation_errors():
    sig = Signal(1, 100e-3, 1000).add_noise()

    with pytest.raises(ValueError, match="block_duration must be > 0"):
        sig.copy().remove_silence(block_duration=0.0)

    with pytest.raises(ValueError, match="overlap_duration must be >= 0"):
        sig.copy().remove_silence(overlap_duration=-1e-3)

    with pytest.raises(
        ValueError,
        match="overlap_duration must be smaller than block_duration",
    ):
        sig.copy().remove_silence(block_duration=10e-3, overlap_duration=10e-3)

    with pytest.raises(ValueError, match="fade_duration must be > 0"):
        sig.copy().remove_silence(fade=True, fade_duration=0.0)


def test_remove_silence_optional_join_fade():
    fs = 1000
    sig = Signal(1, 500e-3, fs)
    sig[50:120] = 1.0
    sig[300:370] = 1.0

    no_fade = sig.copy()
    no_fade.remove_silence(
        threshold_dbfs=-40,
        block_duration=50e-3,
        overlap_duration=10e-3,
    )

    with_fade = sig.copy()
    with_fade.remove_silence(
        threshold_dbfs=-40,
        block_duration=50e-3,
        overlap_duration=10e-3,
        fade=True,
        fade_duration=20e-3,
        win_type="triang",
    )

    assert with_fade.n_samples == no_fade.n_samples
    assert np.sum(np.isclose(with_fade, 1.0)) < np.sum(np.isclose(no_fade, 1.0))


def test_remove_silence_suppresses_expected_analysis_warnings():
    fs = 1000
    sig = Signal(1, 300e-3, fs)
    sig[100:200] = 1.0

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sig.remove_silence(
            threshold_dbfs=-40,
            block_duration=50e-3,
            overlap_duration=10e-3,
        )

    assert len(rec) == 0


def test_concatenate():
    sig_a = Signal(2, 1, 48000).add_noise()
    old_n = sig_a.n_samples
    sig_b = Signal(2, 0.5, 48000).add_noise()

    sig_a.concatenate(sig_b)
    assert sig_a.n_samples == old_n + sig_b.n_samples
    testing.assert_equal(sig_a[old_n:], sig_b)


def test_bandpass_brickwall():
    sig = audio.Signal((2, 2), 1, 48000)
    sig.add_noise().bandpass(500, 100, "brickwall")
    sig = sig.to_freqdomain()
    testing.assert_array_almost_equal(sig[np.abs(sig.freq) > 550], 0)
    testing.assert_array_almost_equal(sig[np.abs(sig.freq) < 450], 0)
    assert np.all(sig[(np.abs(sig.freq) < 550) & (np.abs(sig.freq) > 450)] != 0)


def test_bandpass_gammatone():
    # check real valued output
    sig = audio.Signal(1, 1, 48000).add_tone(500)
    sig2 = sig.copy()
    sig.bandpass(500, 100, "gammatone")
    assert not np.iscomplexobj(sig)
    assert sig.shape == sig2.shape

    # check complex output
    sig = audio.Signal(1, 1, 48000).add_tone(500)
    sig2 = sig.copy()
    with pytest.warns(
        UserWarning,
        match="returns a new Signal instead of modifying in-place",
    ):
        out = sig.bandpass(500, 100, "gammatone", return_complex=True)
    assert out is not sig
    assert np.iscomplexobj(out)
    assert out.shape == sig2.shape
    testing.assert_array_equal(sig, sig2)

    # check equivalence of real and complex results
    sig = audio.Signal(1, 1, 48000).add_tone(500)
    sig2 = sig.copy()
    sig.bandpass(500, 100, "gammatone")
    with pytest.warns(
        UserWarning,
        match="returns a new Signal instead of modifying in-place",
    ):
        out = sig2.bandpass(500, 100, "gammatone", return_complex=True)
    testing.assert_array_equal(out.real, sig)

    # check kwargs
    sig = audio.Signal(1, 1, 48000).add_tone(500)
    out = audio.filter.gammatone(sig, 500, 100, sig2.fs, order=2, attenuation_db=-1)
    sig.bandpass(500, 100, "gammatone", order=2, attenuation_db=-1)
    testing.assert_array_equal(sig, out.real)


def test_bandpass_butterworth():
    sig = audio.Signal(1, 1, 48000).add_noise()

    sig2 = audio.filter.butterworth(sig, 100, 300)
    sig = sig.bandpass(200, 200, "butter")
    testing.assert_array_equal(sig, sig2)


def test_lowpass():
    types = ["brickwall", "butter"]
    f_cut = 300

    for ftype in types:
        sig = audio.Signal((2, 2), 1, 48000).add_uncorr_noise()
        sig2 = audio.filter.lowpass(sig, f_cut, ftype)
        sig.lowpass(f_cut, ftype)
        testing.assert_array_equal(sig, sig2)


def test_highpass():
    types = ["brickwall", "butter"]
    f_cut = 300

    for ftype in types:
        sig = audio.Signal(2, 1, 48000).add_noise()
        sig2 = audio.filter.highpass(sig, f_cut, ftype)
        sig.highpass(f_cut, ftype)
        testing.assert_array_equal(sig, sig2)


def test_channel_indexing():
    sig = Signal((2, 2), 1, 48000).add_noise()
    assert sig.ch[0, 0].shape == (sig.n_samples, 1)
    testing.assert_equal(sig.ch[0, 0][:, 0], sig[:, 0, 0])
    testing.assert_equal(sig.ch[0], sig[:, 0])

    sig = Signal((2, 5), 1, 48000).add_noise()
    assert sig.ch[0].shape == (sig.n_samples, 5)
    assert sig.ch[0, 0].shape == (sig.n_samples, 1)
    testing.assert_equal(sig.ch[0, 0][:, 0], sig[:, 0, 0])
    assert sig.ch[:, 0].shape == (sig.n_samples, 2)
    testing.assert_equal(sig.ch[:, 0], sig[:, :, 0])
    testing.assert_equal(sig.ch[..., 0], sig[:, :, 0])

    sig = Signal(2, 1, 48000)
    sig.ch[0] = 1
    assert np.all(sig[:, 0] == 1)

    sig.ch[1].add_tone(500)
    tone_2 = np.cos(2 * np.pi * sig.time * 500)
    assert sig.ch[1].shape == (sig.n_samples, 1)
    testing.assert_almost_equal(sig.ch[1][:, 0], tone_2)

    # Indexing only one channel should still work
    sig = Signal(1, 1, 40000).add_noise()
    testing.assert_equal(sig.ch[0], sig)
    sig.ch[0] = 1
    testing.assert_equal(sig.ch[0], 1)

    with pytest.raises(IndexError):
        sig.ch[1]

    with pytest.raises(IndexError):
        sig.ch[0, 0]


def test_time_offset():
    sig = Signal(1, 1, 48000)
    assert sig.time_offset == 0
    sig.time_offset = -5
    assert sig.time_offset == -5

    sig2 = sig.copy()
    assert sig2.time_offset == -5


def test_analytical():
    sig = audio.Signal((2, 2), 1, 48000).add_noise()
    asig = sig.to_analytical()
    testing.assert_almost_equal(sig, asig.real)


def test_analytical_tone_quadrature():
    sig = audio.Signal(1, 0.1, 48000).add_tone(500)
    sig2 = audio.Signal(1, 0.1, 48000).add_tone(500, start_phase=-np.pi / 2)

    asig = sig.to_analytical()

    testing.assert_almost_equal(asig.real, sig)
    testing.assert_almost_equal(asig.imag, sig2)


def test_analytical_complex_input_uses_fallback():
    rng = np.random.default_rng(0)
    sig = audio.Signal((2, 3), 0.05, 48000, dtype=complex)
    sig[:] = rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)

    asig = sig.to_analytical()
    ref = sig.to_freqdomain().to_analytical().to_timedomain()

    assert np.iscomplexobj(asig)
    assert asig.shape == sig.shape
    testing.assert_allclose(np.asarray(asig), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_analytical_multidimensional_shape_and_dtype():
    sig = audio.Signal((2, 3, 4), 0.05, 48000).add_noise(seed=0)

    asig = sig.to_analytical()

    assert asig.shape == sig.shape
    assert np.iscomplexobj(asig)
    testing.assert_almost_equal(asig.real, sig)


def test_to_signal():
    rng = np.random.default_rng(0)
    fs = 480000
    sig_array = rng.random(1000)
    sig = audio.as_signal(sig_array, fs)
    assert isinstance(sig, audio.Signal)
    testing.assert_array_equal(sig, sig_array)

    # test multiple dimensions
    fs = 480000
    sig_array = rng.random((1000, 3, 4))
    sig = audio.as_signal(sig_array, fs)
    assert isinstance(sig, audio.Signal)
    testing.assert_array_equal(sig, sig_array)

    # test datatype
    fs = 480000
    sig_array = np.ones([1000], dtype=complex)
    sig = audio.as_signal(sig_array, fs)
    assert isinstance(sig, audio.Signal)
    testing.assert_array_equal(sig, sig_array)


def test_apply_gain():
    sig = audio.Signal(1, 1, 48000).add_noise()
    sig.set_dbfs(-20)
    sig.apply_gain(10)
    testing.assert_almost_equal(sig.stats.dbfs, -10)

    sig.apply_gain(10)
    testing.assert_almost_equal(sig.stats.dbfs, 0)

    sig = audio.Signal(3, 1, 48000).add_uncorr_noise()
    sig.set_dbfs(-20)
    sig.apply_gain(10)
    testing.assert_almost_equal(sig.stats.dbfs, -10)


def test_writefile(tmp_path):
    filename = tmp_path / "test.wav"
    sig = audio.Signal(1, 1, 48000).add_noise()
    sig.write_file(str(filename))
    sig.set_dbfs(-20)
    rsig = audio.from_file(str(filename))
    assert rsig.fs == sig.fs
    assert rsig.shape == sig.shape


@pytest.mark.parametrize(
    ("signal_channels", "kernel_channels", "expected_channels"),
    [
        (1, 1, 1),
        (2, 1, 2),
        (1, 2, 2),
        ((2, 1), 3, (2, 3)),
        ((3, 1), 4, (3, 4)),
        (2, 3, (2, 3)),
        ((2, 4), 3, (2, 4, 3)),
        ((2, 4), (3, 4), (2, 4, 3, 4)),
        (2, 2, 2),
        ((2, 2), (2, 2), (2, 2)),
    ],
)
def test_convolve_shape_cases(signal_channels, kernel_channels, expected_channels):
    _assert_convolution_shape(signal_channels, kernel_channels, expected_channels)


@pytest.mark.parametrize(
    ("signal_channels", "kernel_channels", "expected_channels"),
    [
        (3, (2, 3), (2, 3)),
        ((2, 2), (2, 2, 3), (2, 2, 3)),
        ((1, 3, 3), (3, 3, 4), (1, 3, 3, 4)),
        ((5, 2, 3), (2, 3), (5, 2, 3)),
        ((2, 3), (3, 2), (2, 3, 2)),
    ],
)
def test_convolve_overlap_dimension_cases(
    signal_channels, kernel_channels, expected_channels
):
    _assert_convolution_shape(
        signal_channels,
        kernel_channels,
        expected_channels,
        overlap_dimensions=True,
    )

    # Test channel matching with multiple channels
    sig = audio.Signal((2, 2), 1, 48000).add_noise()
    kernel = audio.Signal((2, 2, 1), 100e-3, 48000)
    sig.convolve(kernel, overlap_dimensions=True)

    fs = 1
    sig = audio.Signal(1, 10, fs)
    sig[:] = 1
    kernel = audio.Signal(1, 1, fs)
    kernel[:] = 2
    sig.convolve(kernel)
    assert np.all(sig == 2)

    fs = 1
    sig = audio.Signal(2, 10, fs)
    sig[:] = 1
    kernel = audio.Signal(2, 1, fs)
    kernel.ch[0] += 1
    kernel.ch[1] += 2
    sig.convolve(kernel)
    assert np.all(sig.ch[0] == 1) & np.all(sig.ch[1] == 2)

    fs = 1
    sig = audio.Signal(2, 10, fs)
    sig[:] = 1
    kernel = audio.Signal(3, 1, fs)
    kernel.ch[0] += 1
    kernel.ch[1] += 2
    sig.convolve(kernel)
    assert np.all(sig.ch[:, 0] == 1) & np.all(sig.ch[:, 1] == 2)

    # Test modes:
    fs = 1
    sig = audio.Signal(2, 10, fs)
    kernel = audio.Signal(3, 5, fs)
    sig.convolve(kernel)
    assert sig.n_samples == 14

    fs = 1
    sig = audio.Signal(2, 10, fs)
    kernel = audio.Signal(3, 5, fs)
    sig.convolve(kernel, mode="same")
    assert sig.n_samples == 10

    fs = 1
    sig = audio.Signal(2, 10, fs)
    kernel = audio.Signal(3, 5, fs)
    sig.convolve(kernel, mode="valid")
    assert sig.n_samples == 6


def test_convolve_accepts_ndarray_kernel():
    # Regression: `kernel` is type-hinted and documented as "Signal or
    # ndarray", but convolve reads kernel.channel_shape / kernel.n_samples,
    # so a plain ndarray raises AttributeError.
    from scipy.signal import fftconvolve

    np.random.seed(0)
    fs = 48000
    sig = audio.Signal(1, 20 / fs, fs)
    sig[:] = np.random.randn(20, 1)
    kernel = np.array([1.0, 0.5, 0.25, 0.125])

    out = sig.convolved(kernel)

    ref = fftconvolve(np.asarray(sig).ravel(), kernel, mode="full")
    testing.assert_allclose(np.asarray(out).ravel(), ref, atol=1e-9)


def test_convolve_complex_kernel_preserves_imaginary():
    # Regression: the output buffer is allocated with dtype=self.dtype, so
    # convolving a real signal with a complex kernel silently discards the
    # imaginary part.
    from scipy.signal import fftconvolve

    np.random.seed(0)
    fs = 48000
    sig = audio.Signal(1, 16 / fs, fs)
    sig[:] = np.random.randn(16, 1)
    kernel = audio.Signal(1, 4 / fs, fs).astype(complex)
    kernel[:] = np.random.randn(4, 1) + 1j * np.random.randn(4, 1)

    out = sig.convolved(kernel)

    assert np.iscomplexobj(np.asarray(out)), (
        "convolving with a complex kernel should produce a complex result"
    )
    ref = fftconvolve(
        np.asarray(sig).ravel(), np.asarray(kernel).ravel(), mode="full"
    )
    testing.assert_allclose(np.asarray(out).ravel(), ref, atol=1e-9)


def test_convolve_singleton_dim_in_overlap():
    # Regression: dim_overlap is computed before the trailing-singleton
    # squeeze, but the reshapes use the post-squeeze dims. When a squeezed
    # dimension took part in the overlap the broadcast fails with a
    # ValueError instead of convolving each channel with the kernel.
    from scipy.signal import fftconvolve

    np.random.seed(0)
    fs = 48000
    sig = audio.Signal((2, 1), 20 / fs, fs)
    sig[:] = np.random.randn(20, 2, 1)
    kernel = audio.Signal(1, 5 / fs, fs)
    kernel[:] = np.random.randn(5, 1)

    out = sig.convolved(kernel)  # must not raise

    sig_arr = np.asarray(sig).reshape(20, 2)
    ker_arr = np.asarray(kernel).ravel()
    ref = np.stack(
        [fftconvolve(sig_arr[:, c], ker_arr, mode="full") for c in range(2)],
        axis=1,
    )
    testing.assert_allclose(np.asarray(out).reshape(ref.shape), ref, atol=1e-9)
