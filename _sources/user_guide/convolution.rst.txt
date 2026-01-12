.. _guide_convolution:

Convolution
===========

Signals can be convolved with a kernel, which is itself another
:class:`audiotoolbox.Signal`. This is commonly used for filtering or to
apply an impulse response to a signal (e.g., a Room Impulse Response or
a Head-Related Impulse Response). The toolbox uses the fast, FFT-based
convolution from ``scipy.signal.fftconvolve``.

The :meth:`audiotoolbox.Signal.convolve` method performs this operation.
Its behavior with multi-dimensional signals can be controlled with the
``overlap_dimensions`` keyword.

Channel-Wise Convolution
------------------------

By default, convolution is performed only along overlapping dimensions
between the signal and the kernel (``overlap_dimensions=True``). This means
that if the channel shapes match, the first channel of the signal is
convolved with the first channel of the kernel, the second with the
second, and so on.

This is useful for applying multi-channel impulse responses to a
multi-channel signal. For example, to simulate a stereo audio signal
being played in a room, you could convolve the 2-channel signal with a
2-channel Room Impulse Response (RIR).

>>> # Assume 'stereo_signal.wav' is a 2-channel audio file
>>> signal = audio.Signal('stereo_signal.wav')
>>>
>>> # Assume 'stereo_rir.wav' is a 2-channel impulse response
>>> rir = audio.Signal('stereo_rir.wav')
>>>
>>> # Convolve the signal with the RIR
>>> signal.convolve(rir)
>>>
>>> # The resulting signal is still 2 channels
>>> signal.n_channels
2

Full Multi-Channel Convolution
------------------------------

If you need to convolve every channel of the signal with every channel
of the kernel, you can set ``overlap_dimensions=False``.

In this case, convolving a two-channel signal with a two-channel kernel
will result in a ``(2, 2)``-shaped channel output, where each element
represents one of the possible signal-kernel convolution pairs.

>>> signal = audio.Signal(n_channels=2, duration=1, fs=48000)
>>> kernel = audio.Signal(n_channels=2, duration=100e-3, fs=48000)
>>> signal.convolve(kernel, overlap_dimensions=False)
>>> signal.n_channels
(2, 2)

Convolution Mode
----------------

The ``mode`` parameter (one of ``{'full', 'valid', 'same'}``) controls
the size of the output signal, corresponding directly to the ``mode``
argument in ``scipy.signal.fftconvolve``. The default is ``'full'``.