'''
Some helper function dealing with statistics that might arise
during psychoacousic experiments
'''

import numpy as np
from scipy.special import binom
from scipy.stats import norm

def logistic_function(x, x0, k, fmin=0, fmax=1):
    ''' Logistic function.

    Evaluates the input data using a Logistic function of the form:
    ..math:: y = \frac{f_{max} - f_{min}}{ 1 + e^{-k(x - x0)}} + f_{min}

    Parameters:
    -----------
    x : scalar or nd_array
      The input values for the logistic function
    x0 : scalar
      The location of the functions midpoint
    k : scalar
      Defines the slope of the function
    fmin : scalar, optional
      The minimum of the function (Default=0)
    fmax : scalar, optional
      The maximum of the function (Default=1)

    Returns:
    --------
    scalar or nd_array
      The evaluated output data
    '''

    exp_factor = -k * (x - x0)
    res =  (fmax - fmin) / (1 + np.exp(exp_factor)) + fmin
    return res


def weibull_function(x, thr, k, fmin=0, fmax=1, fthr=0.75):
    ''' Weibull Psychometric Function

    Parameters:
    -----------
    x : scaler or nd_array
      The input values for the weibull function
    thr : scalar
      The threshold value at which y(thr)=p_thr
    k : scaler
      Defines the slope of the function
    fthr : scaler
      The probability which is to be used at threshold
    fmin : scalar, optional
      The response probability at zero (default = 0.5)
    fmax : scaler, optional
      The maximum probability (default = 1)

    Returns:
    --------
    scalar or nd_array
      The evaluated output data

'''
    l = thr / (np.log((fmin - fmax) / (fthr - fmax)))**(1 / k)
    y = fmax - (fmax - fmin) * np.exp(-(x / l)**k)

    return y



def calc_binom_log_likelyhood(y_i, y_func, n):
    ''' log likelyhood for a binomial distribution

    Parameters:
    -----------
    y_i : scalar or nd_array
      The observed fraction or correct trials
    y_func : scalar or nd_array
      The fraction of sucessfull trials to check against
    n : the number of pres
      the number of observations

    Returns:
    --------
    float : The log likelyhood to observe y_i given a success rate of y_func

    '''
    #calculate the number of occuarnces
    n1 = n * y_i
    # calculate the number of not occurance
    n2 = (1 - y_i) * n
    log_l = np.log(binom(n, n1)  * y_func**(n1) *(1 - y_func)**(n2))
    return np.sum(log_l)

def calc_binom_llh_deviance(yi, model, n):
    ''' Deviance for goodnes of fit estimations

    This returns the Deviance for a binomially distributed parameter

    '''
    llh_saturated = calc_binom_log_likelyhood(yi, yi, n)
    ll_data = calc_binom_log_likelyhood(yi, model, n)
    deviance = 2 * (llh_saturated - ll_data)
    return deviance

def calc_dprime_2afc(pc):
    '''d' value calculated from the percent correct score of a 2AFC

    Calculates the seperability index d' (d prime) from the percent
    correct value of a 2AFC task using the equation

    ..math:: d' = z(P_c)\sqrt{2}

    where z is the inverse of the cumulative distribution function of
    the Gaussian distribution.

    Parameters:
    -----------
    pc : int or nd_array
      The percent correct value

    Returns:
    --------
    int or ndarray: The d' values

    '''

    dprime = norm.ppf(pc) * np.sqrt(2)

    return dprime
