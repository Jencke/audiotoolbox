import audiotools as audio
import numpy as np
import numpy.testing as testing
import pytest

def test_pad_for_fft():
    signal1 = np.ones(100)
    padded = audio.pad_for_fft(signal1)

    #check correct length
    assert len(padded) == 128

    #check zeros in the end and unchanged in beginning
    assert np.array_equal(padded[100:], np.zeros(28))
    assert np.array_equal(padded[:100], signal1)

    signal1 = np.ones([100, 2])
    padded = audio.pad_for_fft(signal1)
    #check correct length
    assert len(padded) == 128

    #check zeros in the end and unchanged in beginning
    assert np.array_equal(padded[100:, :], np.zeros([28, 2]))
    assert np.array_equal(padded[:100, :], signal1)



def test_generate_tone():
    # test frequency, sampling rate and duration
    tone1 = audio.generate_tone(1, 1, 1e3)
    tone2 = audio.generate_tone(2, 0.5, 2e3)
    assert np.array_equal(tone1, tone2)

    #test phaseshift
    tone = audio.generate_tone(1, 1, 1e3, start_phase=np.pi / 2)
    testing.assert_almost_equal(tone[0], 0)
    tone = audio.generate_tone(1, 1, 1e3, start_phase=1 * np.pi)
    testing.assert_almost_equal(tone[0], -1)

def test_get_time():

    tone = audio.generate_tone(1, 1, 1e3)
    time = audio.get_time(tone, 1e3)

    #Test sampling rate
    assert time[2] - time[1] == 1./1e3

    #Test duration
    assert time[-1] == 1 - 1./1e3

    tone1 = audio.generate_tone(1, 1, 1e3)
    tone2 = audio.generate_tone(1, 1, 1e3)

    tone_two_channel = np.column_stack([tone1, tone2])

    time = audio.get_time(tone, 1e3)

    assert len(time) == len(tone_two_channel)

    #Test sampling rate
    assert time[2] - time[1] == 1./1e3

    #Test duration
    assert time[-1] == 1 - 1./1e3

    # Test appearence of extra sample due to numerics
    fs = 48e3
    left = np.linspace(0, 1, 50976)
    time = audio.get_time(left, fs)
    assert len(left) == len(time)

    # Test using integer input value
    fs = 1e3
    tone = audio.generate_tone(1, 1, fs)
    nsamp = len(tone2)
    time1 = audio.get_time(tone, fs)
    time2 = audio.get_time(nsamp, fs)
    testing.assert_equal(time1, time2)

def test_cosine_fade_window():
    window = audio.cosine_fade_window(np.zeros(1000), 100e-3, 1e3)
    n_window = 100

    #test symmentry
    assert np.array_equal(window[:100], window[-100:][::-1])

    #test starts with 0
    assert window[0] == 0

    window = audio.cosine_fade_window(np.zeros(1000), 100e-3, 1e3)
    n_window = 100

    #test if the window is a cosine curve of the right type
    cos_curve = np.concatenate([window[:100], window[-101:]])
    sin = (0.5 * audio.generate_tone(5, 0.2 + 1. / 1e3, 1e3,
                                     start_phase=np.pi)) + 0.5
    testing.assert_array_almost_equal(cos_curve, sin)

    # Test that the last sample in the window is not equal to 1
    nsamp = audio.nsamples(200e-3, 1e3)
    window = audio.cosine_fade_window(np.zeros(nsamp + 1), 100e-3 , 1e3)
    n_window = 100
    assert window[int(nsamp / 2)] == 1
    assert window[int(nsamp / 2 - 1)] != 1
    assert window[int(nsamp / 2 + 1)] != 1
    assert window[int(nsamp / 2 + 1)] == window[int(nsamp / 2 - 1)]

    # Test multichannel window
    window = audio.cosine_fade_window(np.zeros([1000, 2]), 100e-3, 1e3)
    assert np.array_equal(window[:, 0], window[:, 1])
    assert np.array_equal(window[:100, 0], window[-100:, 0][::-1])


def test_gauss_fade_window():
    window = audio.gaussian_fade_window(np.zeros(1000), 100e-3, 1e3)
    n_window = 100

    #test symmentry
    assert np.array_equal(window[:100], window[-100:][::-1])

    #test starts at -60dB
    testing.assert_almost_equal(window[0], 0.001)

    #test setting cutoff
    window = audio.gaussian_fade_window(np.zeros(1000), 100e-3, 1e3, cutoff=-20)
    testing.assert_almost_equal(window[0], 0.1)

    # Test that the last sample in the window is not equal to 1
    nsamp = audio.nsamples(200e-3, 1e3)
    window = audio.gaussian_fade_window(np.zeros(nsamp + 1), 100e-3 , 1e3)
    n_window = 100

    assert window[int(nsamp / 2)] == 1
    assert window[int(nsamp / 2 - 1)] != 1
    assert window[int(nsamp / 2 + 1)] != 1
    assert window[int(nsamp / 2 + 1)] == window[int(nsamp / 2 - 1)]

    # Test multichannel window
    window = audio.gaussian_fade_window(np.zeros([1000, 2]), 100e-3, 1e3)
    assert np.array_equal(window[:, 0], window[:, 1])
    assert np.array_equal(window[:100, 0], window[-100:, 0][::-1])

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


def test_fftshift_signal():
    fs = 48e3
    delay = lambda x: x * 1./fs

    # signal = np.ones(10)
    # sig = audio.fftshift_signal(signal, delay(10), fs, mode='zeros')
    # assert len(sig) == 20
    # testing.assert_allclose(sig[10:], 1)
    # assert np.all(sig[:10] < np.finfo(signal.dtype).resolution)

    signal = np.ones(10)
    signal[-2:] = 0
    sig = audio.fftshift_signal(signal, delay(2), fs)
    assert len(sig) == 10
    assert np.all(sig[:2] < np.finfo(signal.dtype).resolution)
    testing.assert_allclose(sig[2:], 1)

    signal = np.ones(10)
    signal[:2] = 0
    sig = audio.fftshift_signal(signal, delay(-2),fs)
    assert len(sig) == 10
    testing.assert_allclose(sig[:2], 1)
    assert np.all(sig[-2:] < np.finfo(signal.dtype).resolution)


def test_delay_signal():

    signal = audio.generate_tone(1, 1, 1e3, start_phase = 0.5 * np.pi)
    signal += audio.generate_tone(2, 1, 1e3, start_phase = 0.5 * np.pi)

    delayed = audio.delay_signal(signal, 1.5e-3, 1e3)

    phase1 = 1.5e-3 * 1 * 2 * np.pi - 0.5 * np.pi
    phase2 = 1.5e-3 * 2 * 2 * np.pi - 0.5 * np.pi

    shifted = audio.generate_tone(1, 1, 1e3, start_phase=-phase1)
    shifted += audio.generate_tone(2, 1, 1e3, start_phase=-phase2)

    error = np.abs(shifted[:] - delayed[:-2, 1])
    assert np.max(error[10:-10]) <= 1e-3

    # Check if a negative delay results in inverted channels
    delayed_negative = audio.delay_signal(signal, -1.5e-3, 1e3)

    assert np.array_equal(delayed[:, 0], delayed_negative[:, 1])
    assert np.array_equal(delayed[:, 1], delayed_negative[:, 0])

    # Test with noise and full sample shift
    duration = 100e-3
    fs = 48e3
    noise = audio.generate_noise(duration, fs)
    noise *= audio.cosine_fade_window(noise, 20e-3, fs)
    dt = 1. / fs
    delayed = audio.delay_signal(noise, dt * 5, fs)
    testing.assert_almost_equal(delayed[5:, 1], delayed[:-5, 0])

def test_zeropad():
    signal = audio.generate_tone(1, 1, 1e3)

    buffered = audio.zeropad(signal, 10)

    assert len(buffered) - len(signal)  == 20
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
    diff =  np.diff(barks)

    # should be very close to one bark distance
    assert np.round(diff[0], 2) == 1.0

    # check if the array is equally spaced in barks
    testing.assert_array_almost_equal(diff, diff[::-1])

    freqs = audio.freqspace(100, 1200, 22, scale='erb')
    erbs = audio.freq_to_erb(freqs)
    diff = np.diff(erbs)

    # check if really equally spaced in erbs
    testing.assert_array_almost_equal(diff, diff[::-1])


def test_freq_to_erb():
    # test that scale starts with 0
    assert audio.freq_to_erb(0) == 0

    # compare results with original equation
    freq = np.array([100., 1000, 10000])
    nerb =  audio.freq_to_erb(freq)
    nerb2 = (1000 / (24.7 * 4.37)) * np.log(4.37 * (freq / 1000) + 1)
    assert np.array_equal(nerb, nerb2)

def test_freqarange():
    freqs = audio.freqarange(100, 1200, 1, scale='erb')
    erbs = audio.freq_to_erb(freqs)
    diff = np.diff(erbs)
    testing.assert_almost_equal(diff, diff[::-1])

    freqs = audio.freqarange(100, 1200, 0.5, scale='erb')
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

def test_erb_to_freq():
    # Test by inversion from freq_to_erb
    freq = np.array([100., 1000, 10000])
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
    signal = audio.generate_tone(100, 1, fs)
    mod = audio.cos_amp_modulator(signal, 5, fs)
    test = audio.generate_tone(5, 1, fs)
    testing.assert_array_almost_equal(mod, test + 1)
    assert max(mod) == 2.0

    mod = audio.cos_amp_modulator(signal, 5, fs, 0.5)
    assert mod[0] == 1.5

    mod = audio.cos_amp_modulator(signal, 5, fs, start_phase=np.pi)
    test = audio.generate_tone(5, 1, fs, start_phase=np.pi)
    testing.assert_array_almost_equal(mod, test + 1)



def test_calc_dbspl():
    assert audio.calc_dbspl(np.array([20e-6])) == 0
    assert audio.calc_dbspl(np.array([2e-3])) == 40.0

def test_set_dbsl():
    fs = 100e3
    signal = audio.generate_tone(100, 1, fs)
    signal = audio.set_dbspl(signal, 22)
    assert audio.calc_dbspl(signal) == 22

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
    assert(audio.calc_dbfs(signal) == -5)

    # RMS value of a -5 db sine
    m = (10**(-5 / 20)) / np.sqrt(2)

    signal = np.concatenate([-np.ones(10), np.ones(10)])
    signal = np.tile(signal, 100)
    signal = audio.set_dbfs(signal, -5)
    assert(signal.max() == m)


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
    assert(np.round(l_phon, 1) == 30)

    l_pressure = audio.phon_to_dbspl(1238, 78, interpolate=True)
    l_phon = audio.dbspl_to_phon(1238, l_pressure, interpolate=True)
    assert(np.round(l_phon, 1) == 78)


def test_audfilter_bw():

    cf = np.array([200, 1000])
    bws = audio.calc_bandwidth(cf)
    bws2 = 25 + 75 * (1 + 1.4 * (cf / 1000.)**2)**0.69
    assert np.array_equal(bws, bws2)

    bws = audio.calc_bandwidth(cf, 'erb')
    bws2 = 24.7 * (4.37 * (cf/ 1000.) + 1)
    assert np.array_equal(bws, bws2)

    bw = audio.calc_bandwidth(1000.)
    bw2 = audio.calc_bandwidth(1000, 'cbw')

    #default is cbw and type is float for both
    assert type(bw) == type(float())
    assert bw == bw2

    #test that the function also works if providing integer input
    bw = audio.calc_bandwidth(555.)
    bw2 = audio.calc_bandwidth(555)
    assert bw == bw2

    bw = audio.calc_bandwidth(555., 'erb')
    bw2 = audio.calc_bandwidth(555, 'erb')
    assert bw == bw2

def test_generate_noise():
    duration = 1
    fs = 100e3

    noise = audio.generate_noise(duration, fs)
    assert len(noise) == audio.nsamples(duration, fs)
    assert np.abs(noise.mean()) <= 1e-2

    # Test for whole spectrum
    assert np.all(~np.isclose(np.abs(np.fft.fft(noise))[1:], 0))
    # offset has to be zero
    assert np.all(np.isclose(np.abs(np.fft.fft(noise))[0], 0))

    # Test no offset
    testing.assert_almost_equal(noise.mean(), 0)

    # test seed
    noise1 = audio.generate_noise(duration, fs, seed=1)
    noise2 = audio.generate_noise(duration, fs, seed=1)
    noise3 = audio.generate_noise(duration, fs, seed=2)
    testing.assert_equal(noise1, noise2)
    assert ~np.all(noise1 == noise3)



def test_generate_corr_noise():
    from scipy.stats import pearsonr

    duration = 1
    fs = 100e3
    noise = audio.generate_corr_noise(duration, fs)
    noise1 = noise[:, 0]
    noise2 = noise[:, 1]
    power1 = np.mean(noise1**2)
    power2 = np.mean(noise2**2)

    # Test for whole spectrum
    assert np.all(~np.isclose(np.abs(np.fft.fft(noise1)), 0))
    assert np.all(~np.isclose(np.abs(np.fft.fft(noise2)), 0))

    # Test equal Power assumption
    testing.assert_almost_equal(power1, power2)

    # Test orthogonality
    corr_val = []
    for i in range(100):
        noise = audio.generate_corr_noise(duration, fs)
        noise1 = noise[:, 0]
        noise2 = noise[:, 1]
        corr_val.append(pearsonr(noise1, noise2)[0])

    assert np.max(corr_val) < 1e-4
    assert np.median(corr_val) < 1e-6

    # Test definition of covariance
    corr_val = []
    for i in range(100):
        noise = audio.generate_corr_noise(duration, fs, corr=0.5)
        noise1 = noise[:, 0]
        noise2 = noise[:, 1]
        corr_val.append(pearsonr(noise1, noise2)[0] - 0.5)
    assert np.max(corr_val) < 1e-4
    assert np.median(corr_val) < 1e-6


def test_extract_binaural_differences():

    from scipy.signal import hilbert

    # Check phase_difference
    fs = 48000
    signal1 = audio.generate_tone(500, 1, fs)
    signal2 = audio.generate_tone(500, 1, fs, start_phase=0.5 * np.pi)
    ipd, env_diff = audio.extract_binaural_differences(signal1, signal2)

    assert len(ipd) == len(signal1)
    assert np.all(np.isclose(env_diff, 0))
    assert np.all(np.isclose(ipd, -np.pi * 0.5))

    # check log level difference
    signal1 = audio.set_dbspl(audio.generate_tone(500, 1, fs), 50)
    signal2 = audio.set_dbspl(audio.generate_tone(500, 1, fs), 60)
    ipd, env_diff = audio.extract_binaural_differences(signal1, signal2)

    assert np.all(np.isclose(env_diff, -10))

    # check amplitude difference
    fs = 48000
    signal1 = audio.generate_tone(500, 1, fs)
    signal2 = audio.generate_tone(500, 1, fs) * 0.5
    ipd, env_diff = audio.extract_binaural_differences(signal1, signal2,
                                                       log_levels=False)

    assert np.all(np.isclose(env_diff, 0.5))
    assert np.all(np.isclose(ipd, 0))

    # #Test that phase is wrapped to +pi -pi
    # signal = audio.generate_corr_noise(1, fs, corr=0.5)
    # signal1 = signal[:, 0]
    # signal2 = signal[:, 1]
    # n_buf = int(48000 * 100e-3)
    # win = audio.cosine_fade_window(signal1, 100e-3, fs, n_buf)
    # signal1 *= win
    # signal2 *= win
    # ipd, env_diff = audio.extract_binaural_differences(signal[0], signal[1])
    # assert np.max(np.abs(ipd) <= np.pi)


def test_crest_factor():

    # Test that c for sine is equal to sqrt(2)
    signal = audio.generate_tone(100, 1, 100e3)
    c = audio.crest_factor(signal)
    testing.assert_almost_equal(c, 20*np.log10(np.sqrt(2)))

    # test that c for half wave rect. sine is 2
    signal = audio.generate_tone(100, 1, 100e3)
    signal[signal < 0] = 0
    c = audio.crest_factor(signal)
    testing.assert_almost_equal(c, 20*np.log10(2))


def test_phaseshift():
    signal = audio.generate_tone(100, 1, 100e3)
    signal2 = audio.generate_tone(100, 1, 100e3, np.pi)
    signal3 = audio.phase_shift(signal, np.pi, 100e3)
    testing.assert_almost_equal(signal2, signal3)

    signal1 = audio.generate_tone(100, 1, 100e3)
    signal2 = audio.generate_tone(200, 1, 100e3, np.pi)
    signal = np.column_stack([signal1, signal2])
    signal = audio.phase_shift(signal, np.pi, 100e3)


def test_band2rms():

    band = audio.band2rms(50, 1)
    assert band == 50
    band = audio.band2rms(50, 20)
    assert(band == 50 + 10 * np.log10(20))


    band = audio.rms2band(50, 1)
    assert band == 50
    band = audio.rms2band(50, 20)
    assert(band == 50 - 10 * np.log10(20))

def test_crest_factor():
    signal = audio.generate_tone(100, 1, 100e3)
    cfac = audio.crest_factor(signal)
    testing.assert_almost_equal(cfac, np.sqrt(2))

    signal[signal < 0] = 0
    cfac = audio.crest_factor(signal)
    testing.assert_almost_equal(cfac, 2)
