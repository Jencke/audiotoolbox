from audiotools.auditory_filter import audfilter_bw
import audiotools as audio
import numpy as np
import numpy.testing as testing
import pytest

def test_audfilter_bw():

    cf = np.array([200, 1000])
    bws = audfilter_bw(cf)
    bws2 = 25 + 75 * (1 + 1.4 * (cf / 1000)**2)**0.69
    assert np.array_equal(bws, bws2)

    bws = audfilter_bw(cf, 'erb')
    bws2 = 24.7 * (4.37 * (cf/ 1000) + 1)
    assert np.array_equal(bws, bws2)

    bw = audfilter_bw(1000.)
    bw2 = audfilter_bw(1000, 'cbw')

    #default is cbw and type is float for both
    assert type(bw) == type(float())
    assert bw == bw2
