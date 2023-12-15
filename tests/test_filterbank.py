import numpy as np

import audiotools as audio
import audiotools.filter.bank.filterbank as bank
from audiotools.filter import gammatone_filt
from audiotools.filter import butterworth_filt
import numpy.testing as testing
from audiotools.filter.bank import create_filterbank
from audiotools.filter.bank import auditory_gamma_bank, octave_bank
from audiotools.filter.bank.filterbank import FilterBank, ButterworthBank
from audiotools.filter.bank.filterbank import GammaToneBank


def test_base_signal():
    bank = FilterBank([500, 200], [10, 2], 48000, myparam=3)
    assert len(bank) == 2
    assert np.all(bank.params["myparam"] == [3, 3])

    assert isinstance(bank[0], FilterBank)
    assert bank[0].bw == 10
    assert bank[1].fc == 200
    assert bank[0].fs == bank.fs
    assert bank[0].params["myparam"] == 3

    bank = FilterBank(
        np.random.random(10),
        np.random.random(10),
        48000,
        myparam1=np.random.random(10),
        myparam2=np.random.random(10),
    )
    idx_vec = [np.random.randint(0, 10, 3) for i in range(3)]

    for i in idx_vec:
        sub = bank[i]
        np.testing.assert_equal(sub.bw, bank.bw[i])
        np.testing.assert_equal(sub.fs, bank.fs)
        np.testing.assert_equal(sub.fc, bank.fc[i])
        np.testing.assert_equal(sub.params["myparam1"], bank.params["myparam1"][i])
        np.testing.assert_equal(sub.params["myparam2"], bank.params["myparam2"][i])


def test_sub_butterbank():
    bank = create_filterbank(
        fc=np.random.randint(500, 1000, 10),
        bw=np.random.randint(10, 50, 10),
        fs=48000,
        filter_type="butter",
        order=np.random.randint(1, 10, 10),
    )
    idx_vec = [np.random.randint(0, 10, 3) for i in range(3)]

    sig = audio.Signal(1, 1, 48000).add_noise()
    main_out = bank.filt(sig)
    for i in idx_vec:
        sub = bank[i]
        assert isinstance(sub, ButterworthBank)
        np.testing.assert_equal(sub.bw, bank.bw[i])
        np.testing.assert_equal(sub.fs, bank.fs)
        np.testing.assert_equal(sub.fc, bank.fc[i])
        np.testing.assert_equal(sub.params["order"], bank.params["order"][i])
        sub_out = sub.filt(sig)
        np.testing.assert_equal(main_out.ch[i], sub_out)


def test_sub_gamma():
    bank = create_filterbank(
        fc=np.random.randint(500, 1000, 10),
        bw=np.random.randint(10, 50, 10),
        fs=48000,
        filter_type="gammatone",
        order=np.random.randint(1, 10, 10),
    )
    idx_vec = [np.random.randint(0, 10, 3) for i in range(3)]

    sig = audio.Signal(1, 1, 48000).add_noise()
    main_out = bank.filt(sig)
    for i in idx_vec:
        sub = bank[i]
        assert isinstance(sub, GammaToneBank)
        np.testing.assert_equal(sub.bw, bank.bw[i])
        np.testing.assert_equal(sub.fs, bank.fs)
        np.testing.assert_equal(sub.fc, bank.fc[i])
        np.testing.assert_equal(sub.params["order"], bank.params["order"][i])
        sub_out = sub.filt(sig)
        np.testing.assert_equal(main_out.ch[i], sub_out)


def test_create_filterbank():
    fc = [100, 200]
    bw = [10, 20]
    butter = create_filterbank(fc, bw, "butter", 48000)
    assert isinstance(butter, bank.ButterworthBank)
    assert butter.n_filters == 2

    gamma = create_filterbank(fc, bw, "gammatone", 48000)
    assert isinstance(gamma, bank.GammaToneBank)


def test_butterworth():
    fc = [100, 200, 5000]
    bw = [10, 5, 8]
    fs = 48000
    butter = create_filterbank(fc, bw, "butter", fs)

    for i_filt, (fc, bw) in enumerate(zip(fc, bw)):
        low_f = fc - bw / 2
        high_f = fc + bw / 2
        sos = butterworth_filt.design_butterworth(low_f, high_f, fs)
        coeff = butter.coefficents[:, :, i_filt]
        testing.assert_array_equal(sos, coeff)

    # Test filter gain
    sig = audio.Signal(1, 1, 48000)
    fc = np.round(audio.freqarange(100, 4000, 1, "erb"))
    bw = audio.calc_bandwidth(fc, "erb")
    # add a tone at every fc
    for f in fc:
        sig.add_tone(f)
    # create filterbank an run signal
    bank = create_filterbank(fc, bw, "butter", 48000)
    sig_out = bank.filt(sig)
    # check amplitudes at fc of every filter
    amps = np.zeros(len(fc))
    for i_fc, f in enumerate(fc):
        spec = sig_out.ch[i_fc].to_freqdomain()
        amps[i_fc] = np.abs(spec[spec.freq == f])[0]
    # Amplitudes should be 0.5 (two sided spectrum)
    assert np.all((amps - 0.5) <= 0.01)


def test_gammatone():
    # Test if filter coefficents are equivalent tho those of
    # individual filters
    fc = [100, 200, 5000]
    bw = [10, 5, 8]
    fs = 48000
    gamma = create_filterbank(fc, bw, "gammatone", fs)
    for i_filt, (fc, bw) in enumerate(zip(fc, bw)):
        b, a = gammatone_filt.design_gammatone(fc, bw, fs)
        b_bank = gamma.coefficents[0, i_filt]
        a_bank = gamma.coefficents[2:, i_filt]
        testing.assert_array_equal(a, a_bank)
        testing.assert_array_equal(b, b_bank)

    # Test the gain at fc
    sig = audio.Signal(1, 1, 48000)
    fc = np.round(audio.freqarange(100, 4000, 1, "erb"))
    bw = audio.calc_bandwidth(fc, "erb")
    # add a tone at every fc
    for f in fc:
        sig.add_tone(f)
    # create filterbank an run signal
    bank = create_filterbank(fc, bw, "gammatone", 48000)
    sig_out = bank.filt(sig)
    # check amplitudes at fc of every filter
    amps = np.zeros(len(fc))
    for i_fc, f in enumerate(fc):
        spec = sig_out.ch[i_fc].to_freqdomain()
        amps[i_fc] = np.abs(spec[spec.freq == f])[0]
    # Amplitudes hould be 1 (one sided spectrum)
    assert np.all((amps - 1) <= 0.01)

    # Output shape should remain unchanged if filterbank consists of only
    # one filter
    sig = audio.Signal(1, 1, 48000)
    gamma = create_filterbank(500, 79, "gammatone", 48000)
    out = gamma.filt(sig)
    assert sig.shape == out.shape

    # make sure that other dimensions that have only 1 compontent  are kept
    sig = audio.Signal((2, 1), 1, 48000)
    gamma = create_filterbank([100, 200], [79, 80], "gammatone", 48000)
    out = gamma.filt(sig)
    assert out.shape[1:] == (2, 1, 2)


def test_brickwall():
    # Test the gain at fc
    sig = audio.Signal(1, 1, 48000)
    fc = np.round(audio.freqarange(100, 4000, 1, "erb"))
    bw = audio.calc_bandwidth(fc, "erb")
    # add a tone at every fc
    for f in fc:
        sig.add_tone(f)
    # create filterbank an run signal
    bank = create_filterbank(fc, bw, "brickwall", 48000)
    sig_out = bank.filt(sig)
    # check amplitudes at fc of every filter
    amps = np.zeros(len(fc))
    for i_fc, f in enumerate(fc):
        spec = sig_out.ch[i_fc].to_freqdomain()
        amps[i_fc] = np.abs(spec[spec.freq == f])[0]
    # Amplitudes hould be 0.5 (double sided spectrum)
    assert np.all((amps - 0.5) <= 0.01)


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
    assert isinstance(filt_bank, bank.GammaToneBank)

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
