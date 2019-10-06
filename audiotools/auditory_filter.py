import numpy as np

def calc_bandwidth(fc, scale='cbw'):
    '''Calculate auditory filter bandwidth using different scales

    This Function calculates aproximations for the auditory filter
    bandwidth using differnt concepts:

     - cbw: Use the critical bandwidth concept following [1]_
     - erb: Use the equivalent rectangular bandwith concept following [2]_

Parameters:
-----------
fc : float or ndarray
  center frequency in Hz

scale : str
  String indicating the scale that should be used possible values:
  'cbw' or 'erb'. (default='cbw')

    ..[1] Zwicker, E., & Terhardt, E. (1980). Analytical expressions for
          critical-band rate and critical bandwidth as a function of
          frequency. The Journal of the Acoustical Society of America,
          68(5), 1523-1525.

    ..[2] Glasberg, B. R., & Moore, B. C. (1990). Derivation of auditory
          filter shapes from notched-noise data. Hearing Research, 47(1-2),
          103-138.

    '''

    if 'cbw' in scale:
        bw = 25 + 75 * (1 + 1.4 * (fc / 1000)**2)**0.69
    elif 'erb' in scale:
        bw = 24.7 * (4.37 * (fc / 1000) + 1)

    return bw
