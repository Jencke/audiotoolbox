import numpy as np
import pytest

import audiotoolbox as audio
import audiotoolbox.filter.bank.filterbank as fbank_module
from audiotoolbox.filter import gammatone_filt
from audiotoolbox.filter import butterworth_filt
import numpy.testing as testing
from audiotoolbox.filter.bank import create_filterbank
from audiotoolbox.filter.bank import auditory_gamma_bank, octave_bank
from audiotoolbox.filter.bank.filterbank import FilterBank, ButterworthBank
from audiotoolbox.filter.bank.filterbank import GammaToneBank, BrickBank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mono_signal():
    return audio.Signal(1, 1, 48000).add_noise(seed=0)


@pytest.fixture
def stereo_signal():
    return audio.Signal(2, 1, 48000).add_noise(seed=0)


@pytest.fixture
def erb_filterbank():
    fc = np.round(audio.freqarange(100, 4000, 1, "erb"))
    bw = audio.calc_bandwidth(fc, "erb")
    return fc, bw


# ---------------------------------------------------------------------------
# FilterBank base class
# ---------------------------------------------------------------------------


def test_base_signal():
    rng = np.random.default_rng(0)
    b = FilterBank([500, 200], [10, 2], 48000, myparam=3)
    assert len(b) == 2
    assert np.all(b.params["myparam"] == [3, 3])
    assert isinstance(b[0], FilterBank)
    assert b[0].bw == 10
    assert b[1].fc == 200
    assert b[0].fs == b.fs
    assert b[0].params["myparam"] == 3

    b = FilterBank(
        rng.random(10),
        rng.random(10),
        48000,
        myparam1=rng.random(10),
        myparam2=rng.random(10),
    )
    idx_vec = [rng.integers(0, 10, 3) for _ in range(3)]
    for i in idx_vec:
        sub = b[i]
        np.testing.assert_equal(sub.bw, b.bw[i])
        np.testing.assert_equal(sub.fs, b.fs)
        np.testing.assert_equal(sub.fc, b.fc[i])
        np.testing.assert_equal(sub.params["myparam1"], b.params["myparam1"][i])
        np.testing.assert_equal(sub.params["myparam2"], b.params["myparam2"][i])


# ---------------------------------------------------------------------------
# Sub-bank slice + filt consistency
# ---------------------------------------------------------------------------


def test_sub_butterbank(mono_signal):
    rng = np.random.default_rng(0)
    b = create_filterbank(
        fc=rng.integers(500, 1000, 10),
        bw=rng.integers(10, 50, 10),
        fs=48000,
        filter_type="butter",
        order=rng.integers(1, 10, 10),
    )
    idx_vec = [rng.integers(0, 10, 3) for _ in range(3)]
    main_out = b.filt(mono_signal)
    for i in idx_vec:
        sub = b[i]
        assert isinstance(sub, ButterworthBank)
        np.testing.assert_equal(sub.bw, b.bw[i])
        np.testing.assert_equal(sub.fs, b.fs)
        np.testing.assert_equal(sub.fc, b.fc[i])
        np.testing.assert_equal(sub.params["order"], b.params["order"][i])
        sub_out = sub.filt(mono_signal)
        np.testing.assert_equal(main_out.ch[i], sub_out)


def test_sub_gamma(mono_signal):
    rng = np.random.default_rng(0)
    b = create_filterbank(
        fc=rng.integers(500, 1000, 10),
        bw=rng.integers(10, 50, 10),
        fs=48000,
        filter_type="gammatone",
        order=rng.integers(1, 10, 10),
    )
    idx_vec = [rng.integers(0, 10, 3) for _ in range(3)]
    main_out = b.filt(mono_signal)
    for i in idx_vec:
        sub = b[i]
        assert isinstance(sub, GammaToneBank)
        np.testing.assert_equal(sub.bw, b.bw[i])
        np.testing.assert_equal(sub.fs, b.fs)
        np.testing.assert_equal(sub.fc, b.fc[i])
        np.testing.assert_equal(sub.params["order"], b.params["order"][i])
        sub_out = sub.filt(mono_signal)
        np.testing.assert_equal(main_out.ch[i], sub_out)


# ---------------------------------------------------------------------------
# create_filterbank
# ---------------------------------------------------------------------------


def test_create_filterbank():
    fc = [100, 200]
    bw = [10, 20]
    butter = create_filterbank(fc, bw, "butter", 48000)
    assert isinstance(butter, fbank_module.ButterworthBank)
    assert butter.n_filters == 2

    gamma = create_filterbank(fc, bw, "gammatone", 48000)
    assert isinstance(gamma, fbank_module.GammaToneBank)


def test_create_filterbank_unknown_type_raises():
    with pytest.raises((ValueError, Exception)):
        create_filterbank([100], [10], "unknown_type", 48000)


# ---------------------------------------------------------------------------
# Butterworth
# ---------------------------------------------------------------------------


def test_butterworth_coefficients():
    fc_list = [100, 200, 5000]
    bw_list = [10, 5, 8]
    fs = 48000
    butter = create_filterbank(fc_list, bw_list, "butter", fs)

    for i_filt, (f, b) in enumerate(zip(fc_list, bw_list)):
        low_f = f - b / 2
        high_f = f + b / 2
        sos = butterworth_filt.design_butterworth(low_f, high_f, fs)
        coeff = butter.coefficents[:, :, i_filt]
        testing.assert_array_equal(sos, coeff)


def test_butterworth_gain(mono_signal, erb_filterbank):
    fc, bw = erb_filterbank
    sig = audio.Signal(1, 1, 48000)
    for f in fc:
        sig.add_tone(f)
    b = create_filterbank(fc, bw, "butter", 48000)
    sig_out = b.filt(sig)
    amps = np.zeros(len(fc))
    for i_fc, f in enumerate(fc):
        spec = sig_out.ch[i_fc].to_freqdomain()
        amps[i_fc] = np.abs(spec[spec.freq == f]).ravel()[0]
    assert np.all((amps - 0.5) <= 0.01)


def test_butterworth_nd_channels():
    """ButterworthBank with N-dimensional channel input."""
    sig = audio.Signal((3, 2), 0.1, 48000).add_noise(seed=0)
    b = create_filterbank([500, 1000], [50, 100], "butter", 48000)
    out = b.filt(sig)
    assert out.shape[1:] == (3, 2, 2)


# ---------------------------------------------------------------------------
# Gammatone
# ---------------------------------------------------------------------------


def test_gammatone_coefficients():
    fc_list = [100, 200, 5000]
    bw_list = [10, 5, 8]
    fs = 48000
    gamma = create_filterbank(fc_list, bw_list, "gammatone", fs)
    for i_filt, (f, b) in enumerate(zip(fc_list, bw_list)):
        bcoeff, acoeff = gammatone_filt.design_gammatone(f, b, fs)
        b_bank = gamma.coefficents[0, i_filt]
        a_bank = gamma.coefficents[2:, i_filt]
        testing.assert_array_equal(acoeff, a_bank)
        testing.assert_array_equal(bcoeff, b_bank)


def test_gammatone_gain(erb_filterbank):
    fc, bw = erb_filterbank
    sig = audio.Signal(1, 1, 48000)
    for f in fc:
        sig.add_tone(f)
    b = create_filterbank(fc, bw, "gammatone", 48000)
    sig_out = b.filt(sig)
    amps = np.zeros(len(fc))
    for i_fc, f in enumerate(fc):
        spec = sig_out.ch[i_fc].to_freqdomain()
        amps[i_fc] = np.abs(spec[spec.freq == f]).ravel()[0]
    assert np.all((amps - 1) <= 0.01)


def test_gammatone_single_filter_shape():
    sig = audio.Signal(1, 1, 48000)
    gamma = create_filterbank(500, 79, "gammatone", 48000)
    out = gamma.filt(sig)
    assert sig.shape == out.shape


def test_gammatone_nd_channels():
    sig = audio.Signal((2, 1), 1, 48000)
    gamma = create_filterbank([100, 200], [79, 80], "gammatone", 48000)
    out = gamma.filt(sig)
    assert out.shape[1:] == (2, 1, 2)


# ---------------------------------------------------------------------------
# Brickwall
# ---------------------------------------------------------------------------


def test_brickwall_gain(erb_filterbank):
    fc, bw = erb_filterbank
    sig = audio.Signal(1, 1, 48000)
    for f in fc:
        sig.add_tone(f)
    b = create_filterbank(fc, bw, "brickwall", 48000)
    sig_out = b.filt(sig)
    amps = np.zeros(len(fc))
    for i_fc, f in enumerate(fc):
        spec = sig_out.ch[i_fc].to_freqdomain()
        amps[i_fc] = np.abs(spec[spec.freq == f]).ravel()[0]
    assert np.all((amps - 0.5) <= 0.01)


# ---------------------------------------------------------------------------
# Output shape consistency across filter types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filter_type", ["butter", "gammatone", "brickwall"])
def test_output_shape_consistency(filter_type, mono_signal):
    """All three filter types produce the same output shape for identical input."""
    fc = [500, 1000, 2000]
    bw = [50, 100, 200]
    b = create_filterbank(fc, bw, filter_type, 48000)
    out = b.filt(mono_signal)
    assert out.shape[0] == mono_signal.shape[0]
    assert out.shape[-1] == len(fc)


# ---------------------------------------------------------------------------
# Scalar __getitem__ followed by filt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filter_type", ["butter", "gammatone", "brickwall"])
def test_scalar_getitem_filt(filter_type, mono_signal):
    """bank[0].filt(sig) must not raise for any filter type."""
    fc = [500, 1000]
    bw = [50, 100]
    b = create_filterbank(fc, bw, filter_type, 48000)
    sub = b[0]
    # Should produce output without error; shape: same n_samples, single filter
    out = sub.filt(mono_signal)
    assert out.shape[0] == mono_signal.shape[0]


def test_set_params():
    fc = [100, 200, 5000]
    bw = [10, 5, 8]
    fs = 48000
    gamma = create_filterbank(fc, bw, "gammatone", fs, order=5, attenuation_db=-3)

    for i_filt, (fc, bw) in enumerate(zip(fc, bw)):
        b, a = gammatone_filt.design_gammatone(fc, bw, fs, order=5, attenuation_db=-3)
        b_bank = gamma.coefficents[0, i_filt]
        a_bank = gamma.coefficents[2:, i_filt]
        testing.assert_array_equal(a, a_bank)
        testing.assert_array_equal(b, b_bank)


def test_auditory_gamma_bank():
    filt_bank = auditory_gamma_bank(fs=48000)
    assert isinstance(filt_bank, fbank_module.GammaToneBank)

    fcs = audio.freqarange(16, 16000, 1, "erb")
    testing.assert_array_equal(filt_bank.fc, fcs)


def test_butterworth_zero():
    # Catch a fixed bug where filters would oscillate due to wrong inital
    # states
    fs = 48000
    sig = audio.Signal(1, 1, fs)
    fb = create_filterbank([125, 500, 1000], [50, 55, 58], "butter", fs, order=4)
    out = fb.filt(sig)
    testing.assert_allclose(out, 0)


def test_default_octave_bank():
    # Test that power in all channels is aproximatly equal when applyting to
    # white noise
    fs = 48000
    filt_bank = octave_bank(fs)
    noise = audio.Signal(1, 10, 48000).add_noise("pink")
    bank_out = filt_bank.filt(noise)
    power = np.var(bank_out, axis=0)
    assert power.std() < 0.01

    fs = 48000
    filt_bank = octave_bank(fs)
    noise = audio.Signal(2, 20, 48000).add_uncorr_noise(0, ntype="pink")
    bank_out = filt_bank.filt(noise.ch[0])
    power = np.var(bank_out, axis=0)
    assert power.std() < 0.01
    bank_out = filt_bank.filt(noise.ch[1])
    power = np.var(bank_out, axis=0)
    assert power.std() < 0.01

