import scipy.signal as sig
import numpy as np
from .. import core as audio
from typing import Union


import numpy as np
from scipy.signal import lfilter


def efilt(
    signal: audio.Signal, fc: float, bw: float, return_complex: bool = False
) -> audio.Signal:
    """
    Apply complex exponential filter to the signal.

    Parameters:
    -----------
    signal : ndarray
        Input signal.
    fc : float
        Center frequency in Hz.
    bw : float
        Bandwidth in Hz.
    return_complex : bool
        If True, return complex output. If False, return real part only.

    Returns
    -------
    filtered_signal : ndarray
        Filtered output signal.
    """
    b, a = design_efilt(fc, bw, signal.fs)
    filtered_signal = apply_efilt(signal, b, a, return_complex)

    return filtered_signal


def design_efilt(fc: float, bw: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Complex frequency shifted first order lowpass.

    Parameters:
    -----------
    f0 : float
        Center frequency in Hz.
    bw : float
        Bandwidth in Hz.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    b : ndarray
        Numerator coefficients of the filter.
    a : ndarray
        Denominator coefficients of the filter.
    """
    # Convert Hz to Radians per sample
    w0 = 2 * np.pi * fc / fs
    bw_rad = 2 * np.pi * bw / fs

    # Calculate decay factor
    e0 = np.exp(-bw_rad / 2.0)

    # Coefficients
    b = np.array([1 - e0])
    a = np.array([1, -e0 * np.exp(1j * w0)])

    return b, a


def apply_efilt(
    signal: audio.Signal, b: np.ndarray, a: np.ndarray, return_complex: bool = False
) -> audio.Signal:
    """
    Apply complex exponential filter to the signal.

    Parameters:
    -----------
    signal : Signal
        Input signal.
    b : ndarray
        Numerator coefficients of the filter.
    a : ndarray
        Denominator coefficients of the filter.
    return_complex : bool
        If True, return complex output. If False, return real part only.

    Returns
    -------
    filtered_signal : Signal
        Filtered output signal.
    """
    filtered_signal = lfilter(b, a, signal, axis=0)

    if not return_complex:
        filtered_signal = np.real(filtered_signal)

    return filtered_signal
