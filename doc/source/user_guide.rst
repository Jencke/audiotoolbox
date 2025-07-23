**********
User Guide
**********

This user guide is intended to give a quick overview of the main features of audiotoolbox as well as how to use them. For more details, please see the Reference Manual.

Working with Stimuli in the Time Domain
=======================================

audiotoolbox uses the :class:`audiotoolbox.Signal` class to represent stimuli in the time domain. This class provides an easy-to-use method of modifying and analyzing signals.

Creating Signals
----------------

An empty, 1-second long signal with two channels at 48 kHz is initialized by calling:

>>> signal = audio.Signal(n_channels=2, duration=1, fs=48000)

audiotoolbox supports an unlimited number of channels which can also be arranged across multiple dimensions. For example:

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)

By default, modifications are always applied to all channels at the same time. The following two lines thus add 1 to all samples in both channels:

>>> signal = audio.Signal(n_channels=2, duration=1, fs=48000)
>>> signal += 1

Individual channels can easily be addressed by using the :attr:`audiotoolbox.Signal.ch` indexer:

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> signal.ch[0] += 1

This will add 1 only to the first channel. The `ch` indexer also allows for slicing. For example:

>>> signal = audio.Signal(n_channels=3, duration=1, fs=48000)
>>> signal.ch[1:] += 1

This will add 1 to all but the first channel. Internally, the :class:`audiotoolbox.Signal` class is represented as a numpy array where the first dimension is the time axis represented by the number of samples. Channels are then defined by the following dimensions:

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> signal.shape
(48000, 2, 3)

Both the number of samples and the number of channels can be accessed through properties of the :class:`audiotoolbox.Signal` class:

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> print(f'No. of samples: {signal.n_samples}, No. of channels: {signal.n_channels}')
No. of samples: 48000, No. of channels: (2, 3)

The time axis can be directly accessed using the :attr:`audiotoolbox.Signal.time` property:

>>> signal = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> print(signal.time)
[0.00000000e+00 2.08333333e-05 4.16666667e-05 ... 9.99937500e-01 9.99958333e-01 9.99979167e-01]

It's important to understand that all modifications are in-place, meaning that calling a method does not return a changed copy of the signal but directly changes the values of the signal:

>>> signal = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> signal.add_tone(frequency=500)
>>> print(signal.var())
0.49999999999999994

Creating a copy of a Signal requires the explicit use of the :meth:`audiotoolbox.Signal.copy` method. The :meth:`audiotoolbox.Signal.copy_empty` method can be used to create an empty copy with the same shape as the original:

>>> signal = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> signal2 = signal.copy_empty()

Basic Signal Modifications
==========================

Basic signal modifications such as adding a tone or noise are directly available as methods. Tones are easily added through the :meth:`audiotoolbox.Signal.add_tone` method. A signal with two antiphasic 500 Hz tones in the two channels is created by running:

.. plot::
    :include-source:

    sig = audio.Signal(2, 20e-3, 48000)
    sig.ch[0].add_tone(frequency=500, amplitude=1, start_phase=0)
    sig.ch[1].add_tone(frequency=500, amplitude=1, start_phase=np.pi)

    plt.plot(sig.time * 1e3, sig)
    plt.xlabel('Time / ms')
    plt.ylabel('Amplitude')
    plt.title('Antiphasic 500Hz tones')
    plt.show()
    


Fade-in and fade-out ramps with different shapes can be applied using the :meth:`audiotoolbox.Signal.add_fade_window` method:

.. plot::
    :include-source:

    sig = audio.Signal(1, 100e-3, 48000)
    sig.add_tone(frequency=500, amplitude=1, start_phase=0)
    sig.add_fade_window(rise_time=30e-3, type='cos')

    plt.plot(sig.time * 1e3, sig)
    plt.xlabel('Time / ms')
    plt.ylabel('Amplitude')
    plt.title('Tone with raised cos fade-in and -out')
    plt.show()


Similarly, a cosine modulator can be added through the :meth:`audiotoolbox.Signal.add_cos_modulator` method:

.. plot::
    :include-source:

    sig = audio.Signal(1, 500e-3, 48000)
    sig.add_tone(1000)
    sig.add_cos_modulator(frequency=30, m=1)
    sig.add_fade_window(100e-3)

    plt.plot(sig.time * 1e3, sig)
    plt.xlabel('Time / ms')
    plt.ylabel('Amplitude')
    plt.title('1kHz tone with 30Hz modulator')
    plt.show()    

Generating Noise
================

audiotoolbox provides multiple functions to generate noise:

This example adds the same white, pink, or brown Gaussian the signal and plots them as spectrograms (See :meth:`audiotoolbox.oaudio.time_frequency.TimeFrequency`). The noise variance and a seed for the random number generator can be defined by passing the respective argument (see :meth:`audiotoolbox.Signal.add_noise`). 

.. plot::
    :include-source:

    white_noise = audio.Signal(1, 1, 48000).add_noise()
    pink_noise = audio.Signal(1, 1, 48000).add_noise(ntype='pink')
    brown_noise = audio.Signal(1, 1, 48000).add_noise(ntype='brown')

    wspec, fc = white_noise.time_frequency.octave_band_specgram(oct_fraction=3)
    pspec, fc = pink_noise.time_frequency.octave_band_specgram(oct_fraction=3)
    bspec, fc = brown_noise.time_frequency.octave_band_specgram(oct_fraction=3)

    norm = plt.Normalize(min([wspec.min(), pspec.min(), bspec.min()]), max([wspec.max(), pspec.max(), bspec.max()]))
    fig, ax = plt.subplots(2, 2, sharex='all', sharey='all')
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


Uncorrelated noise can be generated using the :meth:`audiotoolbox.Signal.add_uncorr_noise` method. This uses the Gram-Schmidt process to orthogonalize noise tokens to minimize variance in the created correlation:

>>> noise = audio.Signal(3, 1, 48000).add_uncorr_noise(corr=0.2, ntype='white')
>>> np.cov(noise.T)
array([[1.00002083, 0.20000417, 0.20000417],
       [0.20000417, 1.00002083, 0.20000417],
       [0.20000417, 0.20000417, 1.00002083]])

There is also an option to create band-limited, partly-correlated, or uncorrelated noise by defining low-, high-, or band-pass filters that are applied before using the Gram-Schmidt process. For more details, please refer to the documentation of :meth:`audiotoolbox.Signal.add_uncorr_noise`.

.. include:: user_guide/stats.rst

.. include:: user_guide/input_output.rst

.. include:: user_guide/set_level.rst

.. include:: user_guide/time_frequency.rst

.. include:: user_guide/filters.rst
