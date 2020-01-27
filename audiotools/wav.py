import numpy as np
import wave

def readwav(filename, fullscale=True):
    wf = wave.open(filename, 'rb')

    # get information
    nframes = wf.getnframes()
    sampwidth = wf.getsampwidth()
    nchannels = wf.getnchannels()
    fs = wf.getframerate()

    #read whole file
    byte_data = wf.readframes(nframes)

    wf.close()

    # decide on datatype
    if sampwidth == 1:
        dtype = np.int8
        bitdepth = 8
    elif sampwidth == 2:
        dtype = np.int16
        bitdepth = 16

    # convert to int array
    intsignal = np.frombuffer(byte_data, dtype=dtype)
    if nchannels == 2:
        intsignal = intsignal.reshape(nframes, 2)

    #
    if fullscale:
        signal = int_to_fullscale(intsignal, bitdepth)
    else:
        signal = intsignal

    return signal, fs


def int_to_fullscale(signal, bitdepth):
    '''Convert integer to fullscale (64bit float)

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

    '''
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

    fullscalesignal = np.array(signal, dtype=np.float64) / np.iinfo(dtype).max
    return fullscalesignal

def fullscale_to_int(signal, bitdepth):
    '''Convert a fullscale to int (64bit float)

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

    '''
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
    '''Convert a fullscale array into a bytesignal

       Convert a fullscale array into a bytesignal for streaming to soundcard

       Parameters:
       -----------
       signal : ndarray
           The input signal
       bitdepth : int
           The bitdepth of the input signal (8, 16, 32 or 64)

       Returns:
       --------
       ndarray : The bitstream

    '''
    intsignal = fullscale_to_int(signal, bitdepth)
    intsignal = intsignal.reshape(np.product(intsignal.shape))
    bytesignal = intsignal.tostring()

    return bytesignal
