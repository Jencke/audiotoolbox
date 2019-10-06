import audiotools.stats as stats
import numpy as np
import numpy.testing as testing

def test_logistic_function():

    #check unshifted center
    center_out = stats.logistic_function(0, 0, 10)
    assert center_out == 0.5

    #check shifted center
    y = np.array([2.3, 0])
    center_out = stats.logistic_function(y, 2.3, 10)
    assert center_out[0] == 0.5
    assert center_out[1] <= 0.5

    #check max and min with preset parameters
    y = np.array(np.linspace(-60., 60.))
    center_out = stats.logistic_function(y, 0, 10)
    testing.assert_almost_equal(center_out[0], 0)
    assert center_out[-1] == 1

    #check max and min with given parameters
    y = np.array(np.linspace(-60, 60))
    center_out = stats.logistic_function(y, 0, 10, 4, 6)
    assert center_out[0] == 4
    testing.assert_almost_equal(center_out[-1], 6)

    # check center on changed min and max params
    center_out = stats.logistic_function(0, 0, 10, 4, 6)
    assert  center_out == 5

def test_weibull_funciton():

    #check unshifted center
    center_out = stats.logistic_function(0, 0, 10)
    assert center_out == 0.5
