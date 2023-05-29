import audiotools as audio
import numpy as np
import numpy.testing as testing


def test_mean():
    sig = audio.Signal((2, 2), 1, 48000)
    assert sig.stats.mean().shape == sig.n_channels
    assert np.all(sig.stats.mean() == 0)

    sig = audio.Signal((2, 2), 1, 48000)
    sig.ch[0, 1] += 1
    assert sig.stats.mean()[0, 1] == 1


def test_var():
    sig = audio.Signal((2, 2), 1, 48000)
    assert sig.stats.var().shape == sig.n_channels
    assert np.all(sig.stats.var() == 0)

    sig = audio.Signal((2, 2), 1, 48000).add_noise()
    sig *= np.sqrt(2)
    assert sig.stats.var().shape == sig.n_channels
    testing.assert_allclose(sig.stats.var(), 2)


def test_dbspl():
    sig = audio.Signal((2, 2), 1, 48000).add_noise()
    sig *= np.sqrt(2)
    dbspl = audio.calc_dbspl(sig)
    testing.assert_array_almost_equal(sig.stats.dbspl(), dbspl)


def test_dbfs():
    sig = audio.Signal((2, 2), 1, 48000).add_noise()
    sig *= np.sqrt(2)
    dbspl = audio.calc_dbfs(sig)
    testing.assert_array_almost_equal(sig.stats.dbfs(), dbspl)
    assert sig.stats.dbfs().shape == (2, 2)


def test_crest_factor():
    sig = audio.Signal((2, 2), 1, 48000).add_noise()
    sig *= np.sqrt(2)
    comp_val = audio.crest_factor(sig)
    testing.assert_array_almost_equal(sig.stats.crest_factor(), comp_val)
