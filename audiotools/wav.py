import numpy as np
import wave
import soundfile
import audiotools as audio


# file = soundfile.read('../test.wav')

def readwav(filename, fullscale=True):
    """Read a wav file."""

    sig_array, fs = soundfile.read(filename)

    return sig_array, fs


def int_to_fullscale(signal, bitdepth):
    """Convert integer to fullscale (64bit float).

    Converts a integer signal to its 64bit fullscale representation.

    Parameters:
    -----------
    signal : ndarray
        The input signal
    bitdepth : int
        The bitdepth of the input signal (8, 16, 32 or 64)

    Returns:
    --------
    ndarray : The converted signal

    """
    if bitdepth == 8:
        dtype = np.int8
    if bitdepth == 16:
        dtype = np.int16
    if bitdepth == 32:
        dtype = np.int32
    if bitdepth == 64:
        dtype = np.int64

    # only valid if maximum value in bound of bitdepth
    assert np.abs(signal).max() <= np.iinfo(dtype).max

    fullscalesignal = (np.array(signal, dtype=np.float64)
                       / np.iinfo(dtype).max)

    return fullscalesignal


def fullscale_to_int(signal, bitdepth):
    """Convert a fullscale to int (64bit float).

    Converts a fullscale signal into an integer signal of predefined

    Parameters:
    -----------
    signal : ndarray
        The input signal
    bitdepth : int
        The bitdepth of the input signal (8, 16, 32 or 64)

    Returns:
    --------
    ndarray : The converted signal

    """
    if bitdepth == 8:
        dtype = np.int8
    if bitdepth == 16:
        dtype = np.int16
    if bitdepth == 32:
        dtype = np.int32
    if bitdepth == 64:
        dtype = np.int64

    # only valid if maximum smaller 1 (fulsscale)
    assert np.abs(signal).max() <= 1

    # has to be array
    signal = np.array(signal)

    intsignal = np.array(signal * np.iinfo(dtype).max, dtype=dtype)
    return intsignal


def array_to_byte(signal, bitdepth):
    """Convert a fullscale array into a bytesignal.

    Convert a fullscale array into a bytesignal for streaming to
    soundcard

    Parameters:
    -----------
    signal : ndarray
        The input signal
    bitdepth : int
        The bitdepth of the input signal (8, 16, 32 or 64)

    Returns:
    --------
    ndarray : The bitstream

    """
    intsignal = fullscale_to_int(signal, bitdepth)
    intsignal = intsignal.reshape(np.prod(intsignal.shape))
    bytesignal = intsignal.tobytes()

    return bytesignal


def writewav(filename, signal, fs, bitdepth):

    soundfile.write(file=filename,
                    data=signal,
                    samplerate=fs)

    # # get information
    # nchannels = 1 if np.ndim(signal) == 0 else signal.shape[1]

    # if bitdepth % 8 != 0 or bitdepth > 32:
    #     raise ValueError('bitdepth must be a multiple of 8 and < 32')

    # sampwidth = bitdepth // 8

    # bytesignal = array_to_byte(signal, bitdepth)

    # wf = wave.open(filename, 'wb')

    # wf.setnchannels(nchannels)
    # wf.setframerate(fs)
    # wf.setsampwidth(sampwidth)
    # wf.writeframes(bytesignal)

    # wf.close()
