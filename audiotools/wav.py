import numpy as np
import soundfile
import audiotools as audio


def readwav(filename):
    sig, fs = readfile(filename)
    raise (DeprecationWarning("readwav is depricated please use readfile"))
    return sig, fs


def writewav(filename, signal, fs):
    raise (DeprecationWarning("writewav is depricated please use writefile"))
    writefile(filename, signal, fs)


def readfile(filename):
    """Read audiofile using libsndfile.

    Read an audiofile using libsndfile through the soundfile python library.

    Parameters
    ----------
    filename : str
       The path to the file.

    Returns
    -------
    sig_array : np.ndarray
      The data read from the file
    fs : int
      The sampling frequency
    """
    sig_array, fs = soundfile.read(filename)
    return sig_array, fs


def writefile(filename, signal, fs, **kwargs):
    """Write audiofile using libsndfile.

    Write a soundfile using libsndfile through the soundile python library.

    Per default, the major format to be stored is determined by the file
    extension. E.g. a .wav ending indicates a WAV (Microsoft) file. See
    `audiotools.wav.available_formats` for a list of availible formats and
    endings. The major format can be forced by passing a `format` argument.

    If not specifically designed, the subtype (such as Signed 32 bit PCM) is
    chosen as the default for a given format. Avilible subtypes can be checked
    through the `audiotools.wav.available_subtypes` function and forced by
    passing a `subtype` argument.


    Parameters
    ----------
    filename : str
      Filname of the audiofile
    signal : ndarray
      The data
    fs : int
      The sampling rate
    **kwargs :
      Other parameters directly passed to the `soundfile.write` function

    """
    soundfile.write(file=filename, data=signal, samplerate=fs, **kwargs)


def available_formats():
    """Return a dictionary of available major formats."""
    return soundfile.available_formats()


def available_subtypes(format):
    """Return a dictionary of available subtypes.

    Parameters
    ----------
    format : str
        If given, only compatible subtypes are returned.
    """
    return soundfile.available_subtypes(format)
