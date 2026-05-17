import numpy as np
import audiotoolbox as audio

from audiotoolbox.filter.exponential_filter import apply_efilt, design_efilt, efilt


def test_design_efilt_coefficients():
    fc = 1000.0
    bw = 100.0
    fs = 44100.0

    b, a = design_efilt(fc, bw, fs)

    assert len(b) == 1
    assert len(a) == 2
    assert np.iscomplexobj(a)
    assert a[0] == 1.0


def test_apply_efilt_real_vs_complex():
    fs = 100.0
    input_signal = audio.Signal(1, 1, fs).add_tone(10.0)

    b, a = design_efilt(10.0, 5.0, fs)

    out_complex = apply_efilt(input_signal, b, a, return_complex=True)
    assert np.iscomplexobj(out_complex)

    out_real = apply_efilt(input_signal, b, a, return_complex=False)
    assert np.isrealobj(out_real)

    np.testing.assert_allclose(out_real, np.real(out_complex))


def test_efilt_integration():
    fs = 1000.0
    duration = 1.0
    signal_data = audio.Signal(1, duration, fs).add_tone(100.0)

    result = efilt(signal_data, fc=100.0, bw=20.0, return_complex=False)

    assert len(result) == len(signal_data)
    assert np.isrealobj(result)


def test_attenuation_off_center_frequency():
    fs = 10000.0
    fc = 1000.0
    bw = 100.0

    input_signal = audio.Signal(1, 1, fs).add_tone(2000.0)

    b, a = design_efilt(fc, bw, fs)
    output = apply_efilt(input_signal, b, a, return_complex=True)

    mean_amp = np.mean(np.abs(output[1000:]))
    assert mean_amp < 0.2
