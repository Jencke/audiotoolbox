.. toctree::
   :maxdepth: 2

Introduction
============

**audiotoolbox** is a python package designed to generate and anlyze
acoustic stimuli for use in auditory research. It aims to provide an
easy to use and intuitive interface.

Fluent Interface
----------------

The main API of audiotoolbox provides a fluent interface for generating
and analyzing signals. In a fluent interface, methods are applied
in-place and the object itself is returend which allowes methods to be
stacked.

The Signal class
----------------

The audiotoolbox.Signal class is used to work with signals in the time
domain. Like all other classes that are used, the Signal class is
inherited from the numpy.ndarray_ class and thus also inherits all its
methods. It is also directly compatible with most of the packages in
scientific stack such as scipy and matplotlib.

To create a empty signal, the class is called providing the number of
channels, the duration of the stimulus and the sampling rate.

>>> sig = audio.Signal(n_channels=2, duration=1, fs=48000)
>>> print(sig.shape)
(48000, 2)

Basic properties of the signal such as the number of channels, samples
or the duration are availible as properties:

>>> print(sig.n_channels, sig.duration, sig.n_samples, sig.fs)
2 1.0 48000 48000

Signals can have several dimensions:

>>> sig = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> print(sig.shape)
(48000, 2, 3)

to directly index individual channels, the objects provides the `ch` property
which also supports channel slicing

>>> sig = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> slice = sig.ch[0, :]
>>> print(sig.shape, slice.shape)
(48000, 2, 3) (48000, 3)

Methods are allways applied to all channels.

>>> sig = audio.Signal(n_channels=2, duration=1, fs=48000)
>>> sig.add_noise()
>>> np.all(sig.ch[0] == sig.ch[1])
True

thus adds the same noise to both channels of the signal. The ``ch``
indexer if methods should be applied to one individual signal.

>>> sig = audio.Signal(n_channels=2, duration=1, fs=48000)
>>> sig.ch[0].add_noise()
>>> sig.ch[1].add_noise()
>>> np.all(sig.ch[0] == sig.ch[1])
False

Using the ``ch`` indexer is equivalent to direclty indexing the signal

>>> sig = audio.Signal(n_channels=2, duration=1, fs=48000)
>>> sig.ch[0].add_tone(500)
>>> sig[:, 1].add_tone(500)
>>> np.all(sig.ch[0] == sig.ch[1])
True

The FrequencyDomainSignal class
-------------------------------

Audiotools provides a simple mechanism of switching between
time-domain and frequency-domain representation of a signal.

>>> sig = audio.Signal(2, 1, 48000).add_noise()
>>> print(type(sig))
<class 'audiotoolbox.oaudio.signal.Signal'>
>>> fdomain_sig = sig.to_freqdomain()
>>> print(type(fdomain_sig))
<class 'audiotoolbox.oaudio.freqdomain_signal.FrequencyDomainSignal'>

calling the method ``audiotoolbox.Signal.to_freqdomain()`` returns a
FrequencyDomainSignal object which contains the FFT transformed
signal. It is important to note that the object does not directly
contain the FFT transformed but that all frequency components where
normalized by dividing them by the number of samples.

Like the Signal class, the FrequencyDomainSignal is inherits from
``numpy.ndarray`` an empty object can be created using an syntax
identical to creating a Signal object

>>> sig = audio.FrequencyDomainSignal(n_channels=2, duration=1, fs=48000)
>>> print(sig.shape)
(48000, 2)


.. _numpy.ndarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
