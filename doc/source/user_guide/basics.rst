.. _guide_basics:

Signal Basics
=============

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