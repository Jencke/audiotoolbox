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

def test_generate_tone():
    # test frequency, sampling rate and duration
    tone1 = audio.generate_tone(1, 1, 1e3)
    tone2 = audio.generate_tone(2, 0.5, 2e3)
    assert np.array_equal(tone1, tone2)

    # test endpoint
    tone = audio.generate_tone(1, 1, 1e3)
    assert tone[-1] != 0
    tone = audio.generate_tone(1, 1, 1e3, endpoint=True)
    testing.assert_almost_equal(tone[-1], 0)

    #test phaseshift
    tone = audio.generate_tone(1, 1, 1e3, start_phase=np.pi / 2)
    assert tone[0] == 1.0
    tone = audio.generate_tone(1, 1, 1e3, start_phase=1.5 * np.pi)
    assert tone[0] == -1.0

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

def test_cosine_fade_window():
    window = audio.cosine_fade_window(np.zeros(1000), 100e-3, 1e3)
    n_window = 100

    #test symmentry
    assert np.array_equal(window[:100], window[-100:][::-1])

    #test starts with 0
    assert window[0] == 0

    #test if the window is a cosine curve of the right type
    cos_curve = np.concatenate([window[:100], window[-101:]])
    sin = (0.5 * audio.generate_tone(5, 0.2, 1e3, endpoint=True, start_phase=1.5 * np.pi)) + 0.5
    testing.assert_array_almost_equal(cos_curve, sin)


def test_gaus_fade_window():
    window = audio.gaussian_fade_window(np.zeros(1000), 100e-3, 1e3)
    n_window = 100

    #test symmentry
    assert np.array_equal(window[:100], window[-100:][::-1])

    #test starts at -60dB
    testing.assert_almost_equal(window[0], 0.001)

    #test setting cutoff
    window = audio.gaussian_fade_window(np.zeros(1000), 100e-3, 1e3, cutoff=-20)
    testing.assert_almost_equal(window[0], 0.1)

def test_delay_signal():
    signal = audio.generate_tone(1, 1, 1e3)
    signal += audio.generate_tone(2, 1, 1e3)

    delayed = audio.delay_signal(signal, 1.5e-3, 1e3)

    phase1 = 1.5e-3 * 1 * 2 * np.pi
    phase2 = 1.5e-3 * 2 * 2 * np.pi

    shifted = audio.generate_tone(1, 1, 1e3, start_phase=-phase1)
    shifted += audio.generate_tone(2, 1, 1e3, start_phase=-phase2)

    error = np.abs(shifted[:] - delayed[:-2, 1])
    assert np.max(error[10:-10]) <= 1e-3

def test_zero_buffer():
    signal = audio.generate_tone(1, 1, 1e3)

    buffered = audio.zero_buffer(signal, 10)

    assert len(buffered) - len(signal)  == 20
    assert np.array_equal(buffered[:10], buffered[-10:])
    assert np.array_equal(buffered[:10], np.zeros(10))

    buffered = audio.zero_buffer(signal, 0)
    assert len(buffered) == len(signal)

def test_bark():
    # Compare the tabled values to the ones resulting from the equation

    scale = np.array(audio.get_bark_limits()[:-1])
    calc_vals = audio.freq_to_bark(scale)

    assert np.abs(calc_vals - np.arange(len(scale))).max() <= 0.08

    scale = np.array(audio.get_bark_limits())
    calc_vals = audio.freq_to_bark(scale[:-1], True)
    assert np.array_equal(np.arange(0, 24), calc_vals)

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
    assert max(mod) == 2.0

    mod = audio.cos_amp_modulator(signal, 5, fs, 0.5)
    assert mod[0] == 1.5


def test_calc_dbspl():

    assert audio.calc_dbspl(20e-6) == 0
    assert audio.calc_dbspl(2e-3) == 40.0

def test_set_dbsl():
    fs = 100e3
    signal = audio.generate_tone(100, 1, fs)
    signal = audio.set_dbspl(signal, 22)
    assert audio.calc_dbspl(signal) == 22

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
