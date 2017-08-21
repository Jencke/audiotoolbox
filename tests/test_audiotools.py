import audiotools as audio
import numpy as np
import numpy.testing as testing

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

def test_sin_amp_modulate():
    import matplotlib.pyplot as plt
    fs = 10.3e3

    tone = audio.generate_tone(100, 100, fs)
    modulator = audio.sin_amp_modulate(tone, 5, fs, 1)
    signal = modulator * tone
    window = audio.cosine_fade_window(signal, 10e-3, fs)
    padded_sig = audio.pad_for_fft(window * signal)

    freq_amp = np.abs(np.fft.fft(padded_sig))
    freq = np.fft.fftfreq(len(freq_amp), 1/5e3)
    plt.plot(freq, freq_amp / len(freq_amp) * 2)


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
