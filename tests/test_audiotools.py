import audiotools as audio
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
    assert noise.shape == (48000,)

    # test directly using signal
    sig = audio.Signal((2, 3), 1, 48000)
    noise = audio.generate_low_noise_noise(sig, 500, 200, n_rep=10)
    assert noise.shape == (48000, 2, 3)
    testing.assert_array_equal(noise[:, 0, :], noise[:, 1, :])
    testing.assert_array_equal(noise[:, :, 0], noise[:, :, 1])


def test_generate_tone():
    # test frequency, sampling rate and duration
    tone1 = audio.generate_tone(1, 1, 1e3)
    tone2 = audio.generate_tone(0.5, 2, 2e3)
    assert np.array_equal(tone1, tone2)

    # test phaseshift
    tone = audio.generate_tone(1, 1, 1e3, start_phase=np.pi / 2)
    testing.assert_almost_equal(tone[0], 0)
    tone = audio.generate_tone(1, 1, 1e3, start_phase=1 * np.pi)
    testing.assert_almost_equal(tone[0], -1)

    sig = audio.Signal((2, 3), 1, 48000).add_tone(50)
    tone = audio.generate_tone(sig, 50)
    testing.assert_array_equal(sig, tone)


def test_get_time():
    tone = audio.generate_tone(1, 1, 1e3)
    time = audio.get_time(tone, 1e3)

    # Test sampling rate
    assert time[2] - time[1] == 1.0 / 1e3

    # Test duration
    assert time[-1] == 1 - 1.0 / 1e3

    tone1 = audio.generate_tone(1, 1, 1e3)
    tone2 = audio.generate_tone(1, 1, 1e3)

    tone_two_channel = np.column_stack([tone1, tone2])

    time = audio.get_time(tone, 1e3)

    assert len(time) == len(tone_two_channel)

    # Test sampling rate
    assert time[2] - time[1] == 1.0 / 1e3

    # Test duration
    assert time[-1] == 1 - 1.0 / 1e3

    # Test appearence of extra sample due to numerics
    fs = 48e3
    left = np.linspace(0, 1, 50976)
    time = audio.get_time(left, fs)
    assert len(left) == len(time)


def test_cosine_fade_window():
    window = audio.cosine_fade_window(np.zeros(1000), 100e-3, 1e3)

    # test symmentry
    assert np.array_equal(window[:100], window[-100:][::-1])

    # test starts with 0
    assert window[0] == 0

    window = audio.cosine_fade_window(np.zeros(1000), 100e-3, 1e3)

    # test if the window is a cosine curve of the right type
    cos_curve = np.concatenate([window[:100], window[-101:]])
    sin = (0.5 * audio.generate_tone(0.2 + 1.0 / 1e3, 5, 1e3, start_phase=np.pi)) + 0.5
    testing.assert_array_almost_equal(cos_curve, sin)

    # Test that the last sample in the window is not equal to 1
    nsamp = audio.nsamples(200e-3, 1e3)
    window = audio.cosine_fade_window(np.zeros(nsamp + 1), 100e-3, 1e3)
    assert window[int(nsamp / 2)] == 1
    assert window[int(nsamp / 2 - 1)] != 1
    assert window[int(nsamp / 2 + 1)] != 1
    assert window[int(nsamp / 2 + 1)] == window[int(nsamp / 2 - 1)]

    # Test multichannel window
    window = audio.cosine_fade_window(np.zeros([1000, 2]), 100e-3, 1e3)
    assert np.array_equal(window[:, 0], window[:, 1])
    assert np.array_equal(window[:100, 0], window[-100:, 0][::-1])

    sig = audio.Signal((2, 3), 1, 48000)
    win = audio.cosine_fade_window(sig, 100e-3)
    assert win.shape == sig.shape
    testing.assert_array_equal(win[:, 1, 0], win[:, 0, 1])

    # make sure that it also works if the last dimension equals 1
    sig = audio.Signal((2, 1), 1, 48000)
    win = audio.cosine_fade_window(sig, 100e-3)
    assert win.shape == sig.shape
    testing.assert_array_equal(win[:, 1, 0], win[:, 0, 0])


def test_gauss_fade_window():
    window = audio.gaussian_fade_window(np.zeros(1000), 100e-3, 1e3)

    # test symmentry
    assert np.array_equal(window[:100], window[-100:][::-1])

    # test starts at -60dB
    testing.assert_almost_equal(window[0], 0.001)

    # test setting cutoff
    window = audio.gaussian_fade_window(np.zeros(1000), 100e-3, 1e3, cutoff=-20)
    testing.assert_almost_equal(window[0], 0.1)

    # Test that the last sample in the window is not equal to 1
    nsamp = audio.nsamples(200e-3, 1e3)
    window = audio.gaussian_fade_window(np.zeros(nsamp + 1), 100e-3, 1e3)

    assert window[int(nsamp / 2)] == 1
    assert window[int(nsamp / 2 - 1)] != 1
    assert window[int(nsamp / 2 + 1)] != 1
    assert window[int(nsamp / 2 + 1)] == window[int(nsamp / 2 - 1)]

    # Test multichannel window
    window = audio.gaussian_fade_window(np.zeros([1000, 2]), 100e-3, 1e3)
    assert np.array_equal(window[:, 0], window[:, 1])
    assert np.array_equal(window[:100, 0], window[-100:, 0][::-1])

    sig = audio.Signal((2, 3), 1, 48000)
    win = audio.gaussian_fade_window(sig, 100e-3)
    assert win.shape == sig.shape
    testing.assert_array_equal(win[:, 1, 0], win[:, 0, 1])


# def test_shift_signal():

#     signal = np.ones(10)
#     sig = audio.shift_signal(signal, 10, mode='zeros')
#     assert len(sig) == 20
#     assert np.all(sig[10:] == 1)
#     assert np.all(sig[:10] == 0)

#     signal = np.ones(10)
#     signal[-2:] = 0
#     sig = audio.shift_signal(signal, 2, mode='cyclic')
#     assert len(sig) == 10
#     assert np.all(sig[:2] == 0)
#     assert np.all(sig[2:] == 1)

#     signal = np.ones(10)
#     signal[:2] = 0
#     sig = audio.shift_signal(signal, -2, mode='cyclic')
#     assert len(sig) == 10
#     assert np.all(sig[:2] == 1)
#     assert np.all(sig[-2:] == 0)


def test_delay_signal():
    signal = audio.generate_tone(1, 1, 1e3, start_phase=0.5 * np.pi)
    signal += audio.generate_tone(1, 2, 1e3, start_phase=0.5 * np.pi)

    delayed = audio.delay_signal(signal, 1.5e-3, 1e3)

    phase1 = 1.5e-3 * 1 * 2 * np.pi - 0.5 * np.pi
    phase2 = 1.5e-3 * 2 * 2 * np.pi - 0.5 * np.pi

    shifted = audio.generate_tone(1, 1, 1e3, start_phase=-phase1)
    shifted += audio.generate_tone(1, 2, 1e3, start_phase=-phase2)

    error = np.abs(shifted[:] - delayed[:-2, 1])
    assert np.max(error[10:-10]) <= 1e-3

    # Check if a negative delay results in inverted channels
    delayed_negative = audio.delay_signal(signal, -1.5e-3, 1e3)

    assert np.array_equal(delayed[:, 0], delayed_negative[:, 1])
    assert np.array_equal(delayed[:, 1], delayed_negative[:, 0])

    # Test with noise and full sample shift
    duration = 100e-3
    fs = 48e3
    noise = audio.generate_noise(duration, fs=fs)
    noise *= audio.cosine_fade_window(noise, 20e-3, fs)
    dt = 1.0 / fs
    delayed = audio.delay_signal(noise, dt * 5, fs)
    testing.assert_almost_equal(delayed[5:, 1], delayed[:-5, 0])


def test_zeropad():
    signal = audio.generate_tone(1, 1, 1e3)

    buffered = audio.zeropad(signal, 10)

    assert len(buffered) - len(signal) == 20
    assert np.array_equal(buffered[:10], buffered[-10:])
    assert np.array_equal(buffered[:10], np.zeros(10))

    buffered = audio.zeropad(signal, 0)
    assert len(buffered) == len(signal)

    # Test multichannel signal
    signal = audio.generate_tone(1, 1, 1e3)
    mc_signal = np.column_stack([signal, signal])
    mc_buffered = audio.zeropad(mc_signal, 10)
    assert np.array_equal(mc_buffered[:10, 0], mc_buffered[-10:, 1])

    # Test different start and end zeros
    signal = audio.generate_tone(1, 1, 1e3)
    mc_signal = np.column_stack([signal, signal])
    mc_buffered = audio.zeropad(mc_signal, (10, 5))
    assert np.all(mc_buffered[:10] == 0)
    assert np.all(mc_buffered[-5:] == 0)

    sig = audio.Signal(2, 1, 1)
    sig[:] = 1
    zpsig = audio.zeropad(sig, [2, 2])
    assert zpsig.shape == (5, 2)

    sig = audio.Signal((2, 3), 1, 1)
    sig[:] = 1
    zpsig = audio.zeropad(sig, [2, 2])
    assert zpsig.shape == (5, 2, 3)


def test_bark():
    # Compare the tabled values to the ones resulting from the equation

    scale = np.array(audio.get_bark_limits()[:-1])
    calc_vals = audio.freq_to_bark(scale)

    assert np.abs(calc_vals - np.arange(len(scale))).max() <= 0.08

    scale = np.array(audio.get_bark_limits())
    calc_vals = audio.freq_to_bark(scale[:-1], True)
    assert np.array_equal(np.arange(0, 24), calc_vals)


def test_bark_to_freq():
    # test inversion between freq_to_bark and bark_to_freq
    freqs = np.linspace(100, 15e3, 10)
    barks = audio.freq_to_bark(freqs)
    rev_freqs = audio.bark_to_freq(barks)

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


def test_freq_to_erb():
    # test that scale starts with 0
    assert audio.freq_to_erb(0) == 0

    # compare results with original equation
    freq = np.array([100.0, 1000, 10000])
    nerb = audio.freq_to_erb(freq)
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


def test_erb_to_freq():
    # Test by inversion from freq_to_erb
    freq = np.array([100.0, 1000, 10000])
    nerb = audio.freq_to_erb(freq)

    freq2 = audio.erb_to_freq(nerb)
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


def test_cos_amp_modulator():
    fs = 100e3
    signal = audio.generate_tone(1, 100, fs)
    mod = audio.cos_amp_modulator(signal, 5, fs)
    test = audio.generate_tone(1, 5, fs)
    testing.assert_array_almost_equal(mod, test + 1)
    assert max(mod) == 2.0

    mod = audio.cos_amp_modulator(signal, 5, fs, 0.5)
    assert mod[0] == 1.5

    mod = audio.cos_amp_modulator(signal, 5, fs, start_phase=np.pi)
    test = audio.generate_tone(1, 5, fs, start_phase=np.pi)
    testing.assert_array_almost_equal(mod, test + 1)

    sig = audio.Signal(1, 1, 48000).add_tone(5) + 1
    mod = audio.cos_amp_modulator(sig, 5, 1)
    testing.assert_array_equal(sig, mod)

    sig = audio.Signal((2, 3), 1, 48000).add_tone(5) + 1
    mod = audio.cos_amp_modulator(sig, 5, 1)


def test_calc_dbspl():
    assert audio.calc_dbspl(2e-3) == 40
    assert audio.calc_dbspl(20e-6) == 0
    sig = audio.Signal(1, 1, 48000).add_tone(500)
    l_tone = 20 * np.log10(np.sqrt(0.5) / 20e-6)
    assert audio.calc_dbspl(sig) == l_tone


def test_set_dbsl():
    fs = 100e3
    signal = audio.generate_tone(100, 1, fs)
    signal = audio.set_dbspl(signal, 15)
    testing.assert_almost_equal(audio.calc_dbspl(signal), 15)
    assert audio.set_dbspl(1, 0) == 20e-6


def test_calc_dbfs():
    signal = audio.generate_tone(1000, 1, 48000)
    testing.assert_almost_equal(audio.calc_dbfs(signal), 0)

    signal = np.concatenate([-np.ones(10), np.ones(10)])
    signal = np.tile(signal, 100)
    rms_rect = 20 * np.log10(np.sqrt(2))
    testing.assert_almost_equal(audio.calc_dbfs(signal), rms_rect)


def test_set_dbfs():
    signal = audio.generate_tone(1000, 1, 48000)
    signal = audio.set_dbfs(signal, -5)
    testing.assert_almost_equal(audio.calc_dbfs(signal), -5)

    # RMS value of a -5 db sine
    m = (10 ** (-5 / 20)) / np.sqrt(2)

    signal = np.concatenate([-np.ones(10), np.ones(10)])
    signal = np.tile(signal, 100)
    signal = audio.set_dbfs(signal, -5)
    assert signal.max() == m

    assert audio.set_dbfs(2, 0, norm="peak") == 1
    signal = audio.generate_tone(1000, 8, 48000)
    assert audio.set_dbfs(signal, 0, "peak").max() == 1
    assert audio.set_dbfs(signal, -3, "peak").max() == 10 ** (-3 / 20)


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


def test_generate_noise():
    duration = 1
    fs = 100e3

    noise = audio.generate_noise(duration, fs)
    assert len(noise) == audio.nsamples(duration, fs)
    assert np.ndim(noise) == 1
    # Test for whole spectrum
    spec = np.fft.fft(noise)
    assert np.all(~np.isclose(np.abs(spec)[1:], 0))
    testing.assert_almost_equal(np.abs(spec[0]), 0)
    testing.assert_almost_equal(np.var(noise), 1)

    # # Test no offset
    testing.assert_almost_equal(noise.mean(), 0)
    # test seed
    noise1 = audio.generate_noise(duration, fs, seed=1)
    noise2 = audio.generate_noise(duration, fs, seed=1)
    noise3 = audio.generate_noise(duration, fs, seed=2)
    testing.assert_equal(noise1, noise2)
    assert ~np.all(noise1 == noise3)

    # test directly handing over signal
    sig = audio.Signal((2, 3), 1, 10)
    noise = audio.generate_noise(sig)
    assert noise.shape == (10, 2, 3)
    testing.assert_array_equal(noise[:, 0, :], noise[:, 1, :])
    testing.assert_array_equal(noise[:, :, 0], noise[:, :, 1])

    # test directly handing over signal
    noise = audio.generate_noise(1, 10, n_channels=(2, 3))
    assert noise.shape == (10, 2, 3)
    testing.assert_array_equal(noise[:, 0, :], noise[:, 1, :])
    testing.assert_array_equal(noise[:, :, 0], noise[:, :, 1])

    # test multichannel
    noise = audio.generate_noise(1, 10, n_channels=(2, 3), ntype="pink")
    assert noise.shape == (10, 2, 3)
    testing.assert_array_equal(noise[:, 0, :], noise[:, 1, :])
    testing.assert_array_equal(noise[:, :, 0], noise[:, :, 1])


def test_generate_uncorr_noise():
    duration = 1
    fs = 100e3
    noise = audio.generate_uncorr_noise(duration, fs, n_channels=2)
    noise1 = noise[:, 0]
    noise2 = noise[:, 1]
    # Test equal Power assumption
    testing.assert_almost_equal(noise1.var(), noise2.var())

    # Test multichannel
    res_noise = audio.generate_uncorr_noise(1, fs=48000, n_channels=100, corr=0)
    cv = np.corrcoef(res_noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0)

    # Test multichannel
    res_noise = audio.generate_uncorr_noise(1, fs=48000, n_channels=3, corr=0.5)
    cv = np.corrcoef(res_noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5)

    # Test multichannel
    res_noise = audio.generate_uncorr_noise(
        1, fs=48000, n_channels=3, corr=0.5, ntype="pink"
    )
    cv = np.corrcoef(res_noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5)

    # Test vor variance = 1
    noise = audio.generate_uncorr_noise(duration, fs, 2, corr=0.5)
    testing.assert_almost_equal(noise.var(axis=0), 1)

    # Test multiple dimensions:
    noise = audio.generate_uncorr_noise(duration, fs, (2, 3, 4), corr=0.5)
    assert noise.shape[1:] == (2, 3, 4)
    noise = noise.reshape([len(noise), 2 * 3 * 4])
    cv = np.corrcoef(noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5)


def test_generate_uncorr_noise_filter():
    # Test brickwall
    duration = 1
    fs = 100000
    fc = 300
    bw = 200
    bandpass = {"fc": fc, "bw": bw, "filter_type": "brickwall"}
    noise = audio.generate_uncorr_noise(duration, fs, 2, corr=0.5, bandpass=bandpass)
    flow = fc - bw / 2
    fhigh = fc + bw / 2
    spec = np.abs(np.fft.fft(noise, axis=0))
    freqs = np.fft.fftfreq(len(spec), 1.0 / fs)
    passband = (np.abs(freqs) >= flow) & (np.abs(freqs) <= fhigh)
    non_zero = ~np.isclose(spec, 0)
    assert np.array_equal(non_zero[:, 0], passband)
    assert np.array_equal(non_zero[:, 1], passband)

    # test coherence value
    bandpass = {"fc": fc, "bw": bw, "filter_type": "brickwall"}
    noise = audio.generate_uncorr_noise(duration, fs, 4, corr=0.5, bandpass=bandpass)
    cv = np.corrcoef(noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5)

    bandpass = {"fc": fc, "bw": bw, "filter_type": "butter"}
    noise = audio.generate_uncorr_noise(duration, fs, 4, corr=0.5, bandpass=bandpass)
    cv = np.corrcoef(noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5, decimal=6)

    bandpass = {"fc": fc, "bw": bw, "filter_type": "gammatone"}
    noise = audio.generate_uncorr_noise(duration, fs, 4, corr=0.5, bandpass=bandpass)
    cv = np.corrcoef(noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5, decimal=5)

    fcut = 500
    lowpass = {"f_cut": fcut, "filter_type": "brickwall"}
    noise = audio.generate_uncorr_noise(duration, fs, 4, corr=0.5, lowpass=lowpass)
    cv = np.corrcoef(noise.T)
    lower_tri = np.tril(cv, -1)
    lower_tri[lower_tri == 0] = np.nan
    testing.assert_almost_equal(lower_tri[~np.isnan(lower_tri)], 0.5, decimal=5)

    fcut = 500
    highpass = {"f_cut": fcut, "filter_type": "brickwall"}
    noise = audio.generate_uncorr_noise(duration, fs, 4, corr=0.33, highpass=highpass)


def test_extract_binaural_differences():
    from scipy.signal import hilbert

    # Check phase_difference
    fs = 48000
    signal1 = audio.generate_tone(1, 500, fs)
    signal2 = audio.generate_tone(1, 500, fs, start_phase=0.5 * np.pi)
    signal = np.column_stack([signal1, signal2])
    ipd, ild = audio.extract_binaural_differences(signal)

    assert len(ipd) == len(signal1)
    assert np.all(np.isclose(ild, 0))
    assert np.all(np.isclose(ipd, -np.pi * 0.5))

    # check log level difference
    signal1 = audio.set_dbspl(audio.generate_tone(1, 500, fs), 50)
    signal2 = audio.set_dbspl(audio.generate_tone(1, 500, fs), 60)
    signal = np.column_stack([signal1, signal2])
    ipd, ild = audio.extract_binaural_differences(signal)
    assert np.all(np.isclose(ild, -10))

    # check amplitude difference
    fs = 48000
    signal1 = audio.generate_tone(1, 500, fs)
    signal2 = audio.generate_tone(1, 500, fs) * 0.5
    signal = np.column_stack([signal1, signal2])
    ipd, ild = audio.extract_binaural_differences(signal, log_ilds=False)
    assert np.all(np.isclose(ild, 2))
    assert np.all(np.isclose(ipd, 0))


def test_crest_factor():
    # Test that c for sine is equal to sqrt(2)
    signal = audio.generate_tone(100, 1, 100e3)
    c = audio.crest_factor(signal)
    testing.assert_almost_equal(c, 20 * np.log10(np.sqrt(2)))

    # test that c for half wave rect. sine is 2
    signal = audio.generate_tone(100, 1, 100e3)
    signal[signal < 0] = 0
    c = audio.crest_factor(signal)
    testing.assert_almost_equal(c, 20 * np.log10(2))


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
    signal = audio.generate_tone(100, 1, 100e3)
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
        coh[start + 1 : start + nsamp], coh_analytic[: nsamp - 1], rtol=0, atol=0.03
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

    # If the

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
    duration, fs, n_ch = audio.audiotools._duration_is_signal(1, 2, 3)
    assert duration == 1
    assert fs == 2
    assert n_ch == 3

    duration, fs, n_ch = audio.audiotools._duration_is_signal(1, 2)
    assert duration == 1
    assert fs == 2
    assert n_ch == None

    # signal as input
    sig = audio.Signal((2, 3), 1, 2)
    duration, fs, n_ch = audio.audiotools._duration_is_signal(sig)
    assert duration == 1
    assert fs == 2
    assert n_ch == (2, 3)

    # Numpy array as input
    sig = np.zeros((11, 2, 3))
    duration, fs, n_ch = audio.audiotools._duration_is_signal(sig, 3)
    assert duration == 11 / 3
    assert fs == 3
    assert n_ch == (2, 3)


def test_copy_to_ndim():
    a = np.random.random(1000)
    b = audio.audiotools._copy_to_dim(a, (2, 3))
    assert b.shape == (1000, 2, 3)

    b = audio.audiotools._copy_to_dim(a, 3)
    assert b.shape == (1000, 3)
