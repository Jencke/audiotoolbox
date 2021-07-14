import ctypes
from .wav import array_to_byte

PA_CHANNELS_MAX = 32            # Maximum number of channels allowed

# struct definition for the pa_sample_spec struct
class StructPASampleSpec(ctypes.Structure):
    __slots__ = [
        'format',               # The sample format
        'rate',                 # The sample rate
        'channels']             # The number of channels

    _fields_ = [
        ('format', ctypes.c_int),
        ('rate', ctypes.c_uint32),
        ('channels', ctypes.c_uint8)]

# struct definition for the pa_cvolume struct
class StructPAcVolume(ctypes.Structure):
    __slots__ = [
        'channels',             # Number of channels
        'values']               # Per-channel volume

    _fields_ = [
        ('channels', ctypes.c_uint8),
        ('values', ctypes.c_uint32 * PA_CHANNELS_MAX)
    ]

PA_SAMPLE_FORMAT = { 'PA_SAMPLE_U8': 0, # Unsigned 8 Bit PCM
                     'PA_SAMPLE_ALAW': 1, # 8 Bit a-Law
                     'PA_SAMPLE_ULAW': 2,#  8 Bit mu-Law
                     'PA_SAMPLE_S16LE': 3, # Signed 16 Bit PCM, little endian (PC)
                     'PA_SAMPLE_S16BE': 4, # Signed 16 Bit PCM, big endian
                     'PA_SAMPLE_FLOAT32LE': 5, # 32 Bit IEEE floating point, little endian (PC), range -1.0 to 1.0
                     'PA_SAMPLE_FLOAT32BE': 6, # 32 Bit IEEE floating point, big endian, range -1.0 to 1.0
                     'PA_SAMPLE_S32LE': 7, # Signed 32 Bit PCM, little endian (PC)
                     'PA_SAMPLE_S32BE': 8, # Signed 32 Bit PCM, big endian
                     'PA_SAMPLE_S24LE': 9, # Signed 24 Bit PCM packed, little endian (PC). \since 0.9.15
                     'PA_SAMPLE_S24BE': 10, # Signed 24 Bit PCM packed, big endian. \since 0.9.15
                     'PA_SAMPLE_S24_32LE': 11, # Signed 24 Bit PCM in LSB of 32 Bit words, little endian (PC). \since 0.9.15
                     'PA_SAMPLE_S24_32BE': 12 # Signed 24 Bit PCM in LSB of 32 Bit words, big endian. \since 0.9.15
}

PA_STREAM_PLAYBACK = 1


def play(signal, fs, bitdepth, buffsize=1024):  # pragma: no cover
    """Play a sound signal.

    This function plays a sound-signal using pulseaudio.

    """
    byte_signal = array_to_byte(signal, bitdepth)

    if bitdepth == 16:
        format_id = PA_SAMPLE_FORMAT['PA_SAMPLE_S16LE']
    elif bitdepth == 32:
        format_id = PA_SAMPLE_FORMAT['PA_SAMPLE_S32LE']
    else:
        raise Exception('Bitdepth not implemented')

    pa = ctypes.cdll.LoadLibrary('libpulse-simple.so.0')

    sample_spec = StructPASampleSpec()
    sample_spec.rate = fs
    sample_spec.channels = 2
    sample_spec.format = format_id
    error = ctypes.c_int(0)

    s = pa.pa_simple_new(
        None,  # Default server.
        'Python',  # Application's name.
        PA_STREAM_PLAYBACK,  # Stream for playback.
        None,  # Default device.
        'audiotools',  # Stream's description.
        ctypes.byref(sample_spec),  # Sample format.
        None,  # Default channel map.
        None,  # Default buffering attributes.
        ctypes.byref(error)  # Ignore error code.
    )

    if not s:
        raise Exception('Could not create pulse audio stream: {0}!'.format(
            pa.strerror(ctypes.byref(error))))

    nint = buffsize * 2
    n_run = len(byte_signal) // nint
    for i in range(n_run):
        start = i * nint
        end = (i + 1) * nint
        buf = byte_signal[start:end]
        if pa.pa_simple_write(s, buf, len(buf), error):
            raise Exception('Could not play file!')

    if pa.pa_simple_drain(s, error):
        raise Exception('Could not simple drain!')

    pa.pa_simple_free(s)
