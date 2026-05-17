import audiotoolbox as audio
import numpy as np
import numpy.testing as testing
import pytest


def test_pad_for_fft():
    signal1 = np.ones(100)
    padded = audio.pad_for_fft(signal1)

    # check correct length
    assert len(padded) == 128

    # check zeros in the end and unchanged in beginning
    assert np.array_equal(padded[100:], np.zeros(28))
    assert np.array_equal(padded[:100], signal1)

    signal1 = np.ones([100, 2])
    padded = audio.pad_for_fft(signal1)
    # check correct length
    assert len(padded) == 128

    # check zeros in the end and unchanged in beginning
    assert np.array_equal(padded[100:, :], np.zeros([28, 2]))
    assert np.array_equal(padded[:100, :], signal1)


def test_nsamples():
    duration = 1
    fs = 10

    assert audio.nsamples(duration, fs) == 10

    # test for directly using signal class
    sig = audio.Signal(1, 1, 10)
    assert audio.nsamples(sig) == 10


def test_low_noise_noise():
    noise = audio.generate_low_noise_noise(1, 500, 200, fs=48000)
    assert noise.shape == (48000, 1)

    # test directly using signal
    # sig = audio.Signal((2, 3), 1, 48000)
    noise = audio.generate_low_noise_noise(
        duration=1, fc=500, bw=200, n_rep=10, fs=48000, n_channels=(2, 3)
    )
    assert noise.shape == (48000, 2, 3)
    testing.assert_array_equal(noise[:, 0, :], noise[:, 1, :])
    testing.assert_array_equal(noise[:, :, 0], noise[:, :, 1])

def test_bark():
    # Compare the tabled values to the ones resulting from the equation

    scale = np.array(audio.get_bark_limits()[:-1])
    calc_vals = audio.bark.from_freq(scale)

    assert np.abs(calc_vals - np.arange(len(scale))).max() <= 0.08

    scale = np.array(audio.get_bark_limits())
    calc_vals = audio.bark.from_freq(scale[:-1], True)
    assert np.array_equal(np.arange(0, 24), calc_vals)


def test_bark_to_freq():
    # test inversion between freq_to_bark and bark_to_freq
    freqs = np.linspace(100, 15e3, 10)
    barks = audio.bark.from_freq(freqs)
    rev_freqs = audio.bark.to_freq(barks)

    testing.assert_array_almost_equal(freqs, rev_freqs)


def test_freqspace():
    freqs = audio.freqspace(100, 12000, 23)
    barks = audio.freq_to_bark(freqs)
    diff = np.diff(barks)

    # should be very close to one bark distance
    assert np.round(diff[0], 2) == 1.0

    # check if the array is equally spaced in barks
    testing.assert_array_almost_equal(diff, diff[::-1])

    freqs = audio.freqspace(100, 1200, 22, scale="erb")
    erbs = audio.freq_to_erb(freqs)
    diff = np.diff(erbs)

    # check if really equally spaced in erbs
    testing.assert_array_almost_equal(diff, diff[::-1])

    freqs = audio.freqspace(100, 1200, 20, scale="mel")
    mel = audio.mel.from_freq(freqs)
    diff = np.diff(mel)
    testing.assert_array_almost_equal(diff, diff[::-1])

    freqs = audio.freqspace(110, 1760, 20, scale="semitone")
    semi = audio.semitone.from_freq(freqs)
    diff = np.diff(semi)
    testing.assert_array_almost_equal(diff, diff[::-1])

    freqs = audio.freqspace(100, 1200, 20, scale="greenwood")
    green = audio.greenwood.from_freq(freqs)
    diff = np.diff(green)
    testing.assert_array_almost_equal(diff, diff[::-1])


def test_freq_to_erb():
    # test that scale starts with 0
    assert audio.erb.from_freq(0) == 0

    # compare results with original equation
    freq = np.array([100.0, 1000, 10000])
    nerb = audio.erb.from_freq(freq)
    nerb2 = (1000 / (24.7 * 4.37)) * np.log(4.37 * (freq / 1000) + 1)
    assert np.array_equal(nerb, nerb2)


def test_freqarange():
    freqs = audio.freqarange(100, 1200, 1, scale="erb")
    erbs = audio.freq_to_erb(freqs)
    diff = np.diff(erbs)
    testing.assert_almost_equal(diff, diff[::-1])

    freqs = audio.freqarange(100, 1200, 0.5, scale="erb")
    erbs = audio.freq_to_erb(freqs)
    diff = np.diff(erbs)
    testing.assert_almost_equal(diff[0], 0.5)

    freqs = audio.freqarange(100, 1200, 1)
    barks = audio.freq_to_bark(freqs)
    diff = np.diff(barks)
    testing.assert_almost_equal(diff, diff[::-1])

    freqs = audio.freqarange(100, 1200, 0.5)
    barks = audio.freq_to_bark(freqs)
    diff = np.diff(barks)
    testing.assert_almost_equal(diff[0], 0.5)

    freqs = audio.freqarange(16, 16000, 1, "octave")
    assert freqs[-2] == 4000

    freqs = audio.freqarange(16, 16000, 1 / 3, "octave")
    assert freqs[-6] == 4000

    freqs = audio.freqarange(16, 16000, 1 / 2, "octave")
    assert freqs[-4] == 4000

    freqs = audio.freqarange(100, 2000, 1, scale="mel")
    mel = audio.mel.from_freq(freqs)
    diff = np.diff(mel)
    testing.assert_almost_equal(diff[0], 1)

    freqs = audio.freqarange(110, 1760, 1, scale="semitone")
    semi = audio.semitone.from_freq(freqs)
    diff = np.diff(semi)
    testing.assert_almost_equal(diff[0], 1)

    freqs = audio.freqarange(100, 2000, 0.1, scale="greenwood")
    green = audio.greenwood.from_freq(freqs)
    diff = np.diff(green)
    testing.assert_almost_equal(diff[0], 0.1)


def test_erb_to_freq():
    # Test by inversion from freq_to_erb
    freq = np.array([100.0, 1000, 10000])
    nerb = audio.erb.from_freq(freq)

    freq2 = audio.erb.to_freq(nerb)
    np.array_equal(freq2, freq)


def test_time2phase():
    # two simple conversion tests
    f = 1e3
    time = 1e-3
    phase = audio.time2phase(time, f)
    assert phase == (2 * np.pi)

    f = 500
    time = 1e-3
    phase = audio.time2phase(time, f)
    assert phase == (np.pi)


def test_phase2time():
    # simple conversion test
    f = 1e3
    phase = 2 * np.pi
    time = audio.phase2time(phase, f)
    assert time == 1e-3

    # test that phase2time inverts time2phase and that both work on
    # arrays
    f = 1e3
    time = np.linspace(0.1e-3, 1e-3, 100)
    phase = audio.time2phase(time, f)
    calc_time = audio.phase2time(phase, f)

    testing.assert_array_almost_equal(time, calc_time)


def test_phon_to_dbspl():
    # Test some specific Values
    l_pressure = audio.phon_to_dbspl(160, 30)
    assert np.round(l_pressure, 1) == 48.4
    l_pressure = audio.phon_to_dbspl(315, 60)
    assert np.round(l_pressure, 1) == 65.4
    l_pressure = audio.phon_to_dbspl(10000, 80)
    assert np.round(l_pressure, 1) == 91.7

    # Compare interpolated values with default values
    l_int = audio.phon_to_dbspl(10000, 80, interpolate=True)
    l_tab = audio.phon_to_dbspl(10000, 80, interpolate=False)
    testing.assert_almost_equal(l_int, l_tab)

    l_int = audio.phon_to_dbspl(100, 30, interpolate=True)
    l_tab = audio.phon_to_dbspl(100, 30, interpolate=False)
    testing.assert_almost_equal(l_int, l_tab)

    # Test Limits
    with pytest.raises(AssertionError):
        audio.phon_to_dbspl(10000, 90)
        audio.phon_to_dbspl(10000, 10)

    audio.phon_to_dbspl(10000, 10, limit=False)


def test_dbspl_to_phon():
    # Test some specific Values
    l_pressure = audio.phon_to_dbspl(160, 30)
    l_phon = audio.dbspl_to_phon(160, l_pressure)
    assert np.round(l_phon, 1) == 30

    l_pressure = audio.phon_to_dbspl(1238, 78, interpolate=True)
    l_phon = audio.dbspl_to_phon(1238, l_pressure, interpolate=True)
    assert np.round(l_phon, 1) == 78


def test_audfilter_bw():
    cf = np.array([200, 1000])
    bws = audio.calc_bandwidth(cf)
    bws2 = 25 + 75 * (1 + 1.4 * (cf / 1000.0) ** 2) ** 0.69
    assert np.array_equal(bws, bws2)

    bws = audio.calc_bandwidth(cf, "erb")
    bws2 = 24.7 * (4.37 * (cf / 1000.0) + 1)
    assert np.array_equal(bws, bws2)

    bw = audio.calc_bandwidth(1000.0)
    bw2 = audio.calc_bandwidth(1000, "cbw")

    # default is cbw and type is float for both
    assert type(bw) == type(float())
    assert bw == bw2

    # test that the function also works if providing integer input
    bw = audio.calc_bandwidth(555.0)
    bw2 = audio.calc_bandwidth(555)
    assert bw == bw2

    bw = audio.calc_bandwidth(555.0, "erb")
    bw2 = audio.calc_bandwidth(555, "erb")
    assert bw == bw2


def test_deprecated_core_scale_wrappers_warn():
    with pytest.deprecated_call(match="audio.get_bark_limits"):
        audio.get_bark_limits()

    with pytest.deprecated_call(match="audio.bark_to_freq"):
        audio.bark_to_freq(np.array([10.0]))

    with pytest.deprecated_call(match="audio.freq_to_bark"):
        audio.freq_to_bark(np.array([100.0]))

    with pytest.deprecated_call(match="audio.freq_to_erb"):
        audio.freq_to_erb(np.array([100.0]))

    with pytest.deprecated_call(match="audio.erb_to_freq"):
        audio.erb_to_freq(np.array([1.0]))

    with pytest.deprecated_call(match="audio.freq_to_octband"):
        audio.freq_to_octband(1000.0)

    with pytest.deprecated_call(match="audio.octband_to_freq"):
        audio.octband_to_freq(30.0)

    with pytest.deprecated_call(match="audio.calc_bandwidth"):
        audio.calc_bandwidth(1000.0)


def test_extract_binaural_differences():
    from scipy.signal import hilbert

    # Check phase_difference
    fs = 48000
    signal = audio.Signal(2, 1, fs)
    signal.ch[0].add_tone(500)
    signal.ch[1].add_tone(500, start_phase=0.5 * np.pi)
    ipd, ild = audio.extract_binaural_differences(signal)

    assert len(ipd) == len(signal)
    assert np.all(np.isclose(ild, 0))
    assert np.all(np.isclose(ipd, -np.pi * 0.5))

    # check log level difference
    signal = audio.Signal(2, 1, fs)
    signal.ch[0].add_tone(500)
    signal.ch[1].add_tone(500, start_phase=0.5 * np.pi)
    signal.ch[1].apply_gain(10)
    ipd, ild = audio.extract_binaural_differences(signal)
    assert np.all(np.isclose(ild, -10))

    signal = audio.Signal(2, 1, fs)
    signal.ch[0].add_tone(500)
    signal.ch[1].add_tone(500, amplitude=0.5)
    ipd, ild = audio.extract_binaural_differences(signal, log_ilds=False)
    assert np.all(np.isclose(ild, 2))
    assert np.all(np.isclose(ipd, 0))


def test_crest_factor_array():
    # Test that c for sine is equal to sqrt(2)
    signal = audio.Signal(1, 1, 100000).add_tone(100)
    c = audio.crest_factor(signal)
    testing.assert_almost_equal(c, np.sqrt(2))

    # test that c for half wave rect. sine is 2
    signal = audio.Signal(1, 1, 100000).add_tone(100)
    signal[signal < 0] = 0
    c = audio.crest_factor(signal)
    testing.assert_almost_equal(c, 2)


def test_band2rms():
    band = audio.band2rms(50, 1)
    assert band == 50
    band = audio.band2rms(50, 20)
    assert band == 50 + 10 * np.log10(20)

    band = audio.rms2band(50, 1)
    assert band == 50
    band = audio.rms2band(50, 20)
    assert band == 50 - 10 * np.log10(20)


def test_crest_factor():
    signal = audio.Signal(1, 1, 48000).add_tone(1000)
    cfac = audio.crest_factor(signal)
    testing.assert_almost_equal(cfac, np.sqrt(2))


def test_cmplx_crosscorr():
    cf = 500
    bw = 100
    sig = audio.Signal(2, 100, 48000).add_noise()
    sig.bandpass(cf, bw, "brickwall")
    coh = audio.cmplx_crosscorr(sig)

    # Analytic coherence for aboves signal
    coh_analytic = (
        np.sin(np.pi * bw * sig.time[1:])
        / (np.pi * bw * sig.time[1:])
        * np.exp(1j * 2 * np.pi * cf * sig.time[1:])
    )

    assert isinstance(coh, audio.Signal)
    testing.assert_almost_equal(np.abs(coh[coh.time == 0]), 1)
    nsamp = 1000
    start = np.where(coh.time == 0)[0][0]
    testing.assert_allclose(
        coh[start + 1 : start + nsamp, 0], coh_analytic[: nsamp - 1], rtol=0, atol=0.03
    )

    # calculate auto-coherrence
    coh2 = audio.cmplx_crosscorr(sig.ch[0])
    testing.assert_array_equal(coh, coh2)

    # test using numpy arrays
    sig = np.asarray(sig)
    coh3 = audio.cmplx_crosscorr(sig)
    testing.assert_array_equal(coh3, coh)

    cf = 500
    bw = 100
    sig = audio.Signal(2, 100, 48000).add_uncorr_noise(0.5)
    sig.bandpass(cf, bw, "brickwall")
    coh = audio.cmplx_crosscorr(sig)
    testing.assert_allclose(coh.abs()[coh.time == 0], 0.5, rtol=0.05)


def test_cmplx_correlation():
    signal = audio.Signal(1, 1, 48000)
    with pytest.raises(ValueError):
        audio.cmplx_corr(signal)
    signal = audio.Signal(3, 1, 48000)
    with pytest.raises(ValueError):
        audio.cmplx_corr(signal)

    signal = audio.Signal(2, 1, 48000).add_noise()
    ccc = complex(audio.cmplx_corr(signal))
    testing.assert_allclose(np.abs(ccc), 1)
    testing.assert_almost_equal(np.angle(ccc), 0)

    signal = audio.Signal(2, 1, 48000).add_noise()
    signal *= 5.2
    ccc = complex(audio.cmplx_corr(signal))
    testing.assert_allclose(np.abs(ccc), 1)

    signal = audio.Signal(2, 1, 48000).add_noise()
    signal.lowpass(20000, "brickwall")
    signal.ch[1].phase_shift(np.pi / 2)
    ccc = audio.cmplx_corr(signal)
    testing.assert_allclose(np.angle(ccc), np.pi / 2)

    signal = audio.Signal(2, 1, 48000).add_uncorr_noise(0.2)
    ccc = audio.cmplx_corr(signal)
    testing.assert_allclose(np.abs(ccc), 0.2, atol=0.001)
    testing.assert_allclose(np.angle(ccc), 0, atol=0.1)

    signal = audio.Signal((2, 3, 2), 1, 48000).add_uncorr_noise(0.2)
    ccc = audio.cmplx_corr(signal)
    testing.assert_allclose(np.abs(ccc), 0.2, atol=0.001)
    testing.assert_allclose(np.angle(ccc), 0, atol=0.1)


def test_duration_is_signal():
    # direct input
    duration, fs, n_ch = audio.core._duration_is_signal(1, 2, 3)
    assert duration == 1
    assert fs == 2
    assert n_ch == 3

    duration, fs, n_ch = audio.core._duration_is_signal(1, 2)
    assert duration == 1
    assert fs == 2
    assert n_ch == None

    # signal as input
    sig = audio.Signal((2, 3), 1, 2)
    duration, fs, n_ch = audio.core._duration_is_signal(sig)
    assert duration == 1
    assert fs == 2
    assert n_ch == (2, 3)

    # Numpy array as input
    sig = np.zeros((11, 2, 3))
    duration, fs, n_ch = audio.core._duration_is_signal(sig, 3)
    assert duration == 11 / 3
    assert fs == 3
    assert n_ch == (2, 3)


def test_copy_to_ndim():
    a = np.random.random(1000)
    b = audio.core._copy_to_dim(a, (2, 3))
    assert b.shape == (1000, 2, 3)

    b = audio.core._copy_to_dim(a, 3)
    assert b.shape == (1000, 3)


def test_crossfade():
    # cosine fade between uncorrelated noise should keep equal variance
    sig1 = audio.Signal((2, 10), 1, 48000).add_noise()
    sig2 = audio.Signal((2, 10), 1, 48000).add_noise()
    out = audio.crossfade(sig1, sig2, 450e-3, fade_type="cos")
    assert np.abs(1 - out.stats.var.mean()) < 0.01

    # linear fade between uncorrelated noise should decrease variance
    sig1 = audio.Signal((2, 10), 1, 48000).add_noise()
    sig2 = audio.Signal((2, 10), 1, 48000).add_noise()
    out = audio.crossfade(sig1, sig2, 1, fade_type="linear")
    assert np.abs(1 - out.stats.var.mean()) > 0.1

    sig1 = audio.Signal(1, 1, 48000).add_noise()
    sig2 = audio.Signal(1, 1, 48000).add_noise()
    out = audio.crossfade(sig1, sig2, 450e-3, fade_type="cos")
    assert np.abs(1 - out.stats.var.mean()) < 0.01


def test_inst_cmplx_corr_magnitude_bounded():
    """|inst_cmplx_corr| must be <= 1 for all samples (coherence property)."""
    sig = audio.Signal(2, 0.5, 48000).add_noise(seed=0)
    coh = audio.inst_cmplx_corr(sig, window_duration=10e-3)
    assert np.all(np.abs(coh) <= 1.0 + 1e-9)
