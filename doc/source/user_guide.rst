**********
User Guide
**********

This user guide is intended to give a quick overview of the main features of
**audiotoolbox**, as well as how to use them. For more details, please see
the Reference Manual.

Working with Stimuli in the Time Domain
=======================================

**audiotoolbox** uses the :class:`audiotoolbox.Signal` class to represent
stimuli in the time domain. This class provides an easy-to-use method for
modifying and analyzing signals.

Creating Signals
----------------

An empty, 1-second long signal with two channels at 48 kHz is initialized
by calling:

>>> import audiotoolbox as audio
>>> import numpy as np
>>>
>>> signal = audio.Signal(n_channels=2, duration=1, fs=48000)

**audiotoolbox** supports an unlimited number of channels, which can also be
arranged across multiple dimensions. For example:

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)

By default, modifications are applied to all channels simultaneously.
The following two lines add 1 to all samples in all channels:

>>> signal = audio.Signal(n_channels=2, duration=1, fs=48000)
>>> signal += 1

Individual channels can be addressed easily using the
:attr:`audiotoolbox.Signal.ch` indexer:

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> signal.ch[0] += 1

This will add 1 only to the first channel group. The ``ch`` indexer also
allows for slicing:

>>> signal = audio.Signal(n_channels=3, duration=1, fs=48000)
>>> signal.ch[1:] += 1

This will add 1 to all but the first channel. Internally, the
:class:`audiotoolbox.Signal` class is a ``numpy.ndarray`` where the first
dimension is the time axis (number of samples). The subsequent dimensions
define the channels:

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> signal.shape
(48000, 2, 3)

The number of samples and the number of channels can be accessed through
properties of the :class:`audiotoolbox.Signal` class:

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> print(f'No. of samples: {signal.n_samples}, No. of channels: {signal.n_channels}')
No. of samples: 48000, No. of channels: (2, 3)

The time axis can be accessed directly using the
:attr:`audiotoolbox.Signal.time` property:

>>> signal = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> signal.time
array([0.00000000e+00, 2.08333333e-05, 4.16666667e-05, ...,
       9.99937500e-01, 9.99958333e-01, 9.99979167e-01])

It's important to understand that all modifications are in-place, meaning
that calling a method does not return a changed copy of the signal but
directly changes the signal's data:

>>> signal = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> signal.add_tone(frequency=500)
>>> signal.var()
0.49999999999999994

Creating a copy of a Signal requires the explicit use of the
:meth:`audiotoolbox.Signal.copy` method. The
:meth:`audiotoolbox.Signal.copy_empty` method can be used to create an
empty copy with the same shape as the original:

>>> signal = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> signal2 = signal.copy_empty()

Basic Signal Modifications
==========================

Basic signal modifications, such as adding a tone or noise, are directly
available as methods. Tones are easily added through the
:meth:`audiotoolbox.Signal.add_tone` method. A signal with two antiphasic
500 Hz tones in its two channels is created by running:

.. plot::
   :include-source:

   import audiotoolbox as audio
   import numpy as np
   import matplotlib.pyplot as plt

   sig = audio.Signal(n_channels=2, duration=20e-3, fs=48000)
   sig.ch[0].add_tone(frequency=500, amplitude=1, start_phase=0)
   sig.ch[1].add_tone(frequency=500, amplitude=1, start_phase=np.pi)

   plt.plot(sig.time * 1e3, sig)
   plt.xlabel('Time / ms')
   plt.ylabel('Amplitude')
   plt.title('Antiphasic 500Hz Tones')
   plt.grid(True)
   plt.show()

Fade-in and fade-out ramps with different shapes can be applied using the
:meth:`audiotoolbox.Signal.add_fade_window` method:

.. plot::
   :include-source:

   import audiotoolbox as audio
   import matplotlib.pyplot as plt

   sig = audio.Signal(n_channels=1, duration=100e-3, fs=48000)
   sig.add_tone(frequency=500, amplitude=1, start_phase=0)
   sig.add_fade_window(rise_time=30e-3, type='cos')

   plt.plot(sig.time * 1e3, sig)
   plt.xlabel('Time / ms')
   plt.ylabel('Amplitude')
   plt.title('Tone with Raised Cosine Fade-in and -out')
   plt.grid(True)
   plt.show()

Similarly, a cosine modulator can be added through the
:meth:`audiotoolbox.Signal.add_cos_modulator` method:

.. plot::
   :include-source:

   import audiotoolbox as audio
   import matplotlib.pyplot as plt

   sig = audio.Signal(n_channels=1, duration=500e-3, fs=48000)
   sig.add_tone(1000)
   sig.add_cos_modulator(frequency=30, m=1)
   sig.add_fade_window(100e-3)

   plt.plot(sig.time * 1e3, sig)
   plt.xlabel('Time / ms')
   plt.ylabel('Amplitude')
   plt.title('1kHz Tone with 30Hz Modulator')
   plt.grid(True)
   plt.show()

Generating Noise
================

**audiotoolbox** provides multiple functions to generate noise. This example
adds white, pink, and brown Gaussian noise to a signal and plots their
spectrograms. The noise variance and a seed for the random number
generator can be defined by passing the respective arguments (see
:meth:`audiotoolbox.Signal.add_noise`).

.. plot::
   :include-source:

   import audiotoolbox as audio
   import matplotlib.pyplot as plt

   white_noise = audio.Signal(1, 1, 48000).add_noise()
   pink_noise = audio.Signal(1, 1, 48000).add_noise(ntype='pink')
   brown_noise = audio.Signal(1, 1, 48000).add_noise(ntype='brown')

   wspec, fc = white_noise.time_frequency.octave_band_specgram(oct_fraction=3)
   pspec, fc = pink_noise.time_frequency.octave_band_specgram(oct_fraction=3)
   bspec, fc = brown_noise.time_frequency.octave_band_specgram(oct_fraction=3)

   norm = plt.Normalize(
       vmin=min([wspec.min(), pspec.min(), bspec.min()]),
       vmax=max([wspec.max(), pspec.max(), bspec.max()])
   )
   fig, ax = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(8, 8))
   ax[0, 0].set_title('White Noise')
   ax[0, 0].pcolormesh(wspec.time, fc, wspec.T, norm=norm)
   ax[0, 1].set_title('Pink Noise')
   ax[0, 1].pcolormesh(pspec.time, fc, pspec.T, norm=norm)
   ax[1, 0].set_title('Brown Noise')
   ax[1, 0].pcolormesh(bspec.time, fc, bspec.T, norm=norm)

   ax[1, 0].set_xlabel("Time / s")
   for a in ax[:, 0]:
       a.set_ylabel('Frequency / Hz')

   for a in ax.flatten():
       a.set_yscale('log')
   ax[1, 1].set_visible(False)
   plt.tight_layout()
   plt.show()

Uncorrelated noise can be generated using the
:meth:`audiotoolbox.Signal.add_uncorr_noise` method. This uses the
Gram-Schmidt process to orthogonalize noise tokens to minimize variance
in the created correlation:

>>> noise = audio.Signal(3, 1, 48000).add_uncorr_noise(corr=0.2, ntype='white')
>>> np.cov(noise.T)
array([[1.00002083, 0.20000417, 0.20000417],
       [0.20000417, 1.00002083, 0.20000417],
       [0.20000417, 0.20000417, 1.00002083]])

There is also an option to create band-limited, partly-correlated, or
uncorrelated noise by defining low-, high-, or band-pass filters that are
applied before the Gram-Schmidt process. For more details, please refer
to the documentation of :meth:`audiotoolbox.Signal.add_uncorr_noise`.

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


.. include:: user_guide/stats.rst
.. include:: user_guide/input_output.rst
.. include:: user_guide/set_level.rst
.. include:: user_guide/time_frequency.rst
.. include:: user_guide/filters.rst
