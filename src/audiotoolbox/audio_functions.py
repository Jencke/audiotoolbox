import numpy as np
from numpy import pi


def cos_amp_modulator(duration, modulator_freq, fs=None, mod_index=1, start_phase=0):
    r"""Cosinus amplitude modulator.

    Returns a cosinus amplitude modulator following the equation:

    ..  math:: 1 + m \cos{2 \pi f_m t \phi_{0}}

    where :math:`m` is the modulation depth, :math:`f_m` is the
    modualtion frequency and :math:`t` is the time.  :math;`\phi_0` is
    the start phase

    Parameters
    ----------
    duration : ndarray An input array that is used to determine the
    length of the modulator.

    modulator_freq : float The frequency of the cosine modulator.

    fs : float The sample frequency of the input signal.

    mod_index: float, optional The modulation index.  (Default = 1)

    Returns
    -------
    ndarray : The modulator

    See Also
    --------

    audiotoolbox.Signal.add_cos_modulator
    """
    duration, fs, n_channels = _duration_is_signal(duration, fs)

    time = get_time(duration, fs)

    modulator = 1 + mod_index * np.cos(2 * pi * modulator_freq * time + start_phase)

    modulator = _copy_to_dim(modulator, n_channels)

    return modulator
