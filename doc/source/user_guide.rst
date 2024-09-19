**********
User Guide
**********

This user guide is intended to give a quick overview over the main features of audiotoolbox as well as how to use them. for more details please see the Reference Manual.

Working with Stimuli in the time domain
=======================================

audiotools uses the :meth:`audiotools.Signal` class to represent stimuli in the time domain. This class provides an easy to use method of modifying and analyzing signals.

An empty, 1 second long signal with two channels at 48kHz is initialized by calling:

>>> signal = audio.Signal(n_channels=2, duration=1, fs=48000)

audiotools supports an unlimited number of channels which can also be arranged across multiple dimensions. E.g.

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)

Per default, modifications are always applied to all channels at the same time. The following two lines thus add 1 to all samples in both channels.

>>> signal = audio.Signal(n_channels=2, duration=1, fs=48000)
>>> signal += 1

Individual channels can easily be addressed by using the :attr:`audiotools.Signal.ch` indexer.

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> signal.ch[0] += 1

will thus add 1 only to the first channel. The `ch` indexer also allows for slicing. E.g.

>>> signal = audio.Signal(n_channels=3, duration=1, fs=48000)
>>> signal.ch[1:] += 1

This will only add 1 to all but the first channel. Internally, the :meth:`audiotools.Signal` class is represented as a numpy array where the first dimension is the time axis represented by the number of samples. Channels are then defined by the following dimensions.

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> signal.shape()
(48000, 2, 3)

Both the number of samples and the number of channels can be accessed through properties of the :meth:`audiotools.Signal` class:

>>> signal = audio.Signal(n_channels=(2, 3), duration=1, fs=48000)
>>> print(f'No. of samples: {signal.n_samples}, No. of channels: {signal.n_channels})
No. of samples: 48000, No. of channels: (2, 3)

The Time axis can be directly accessed using the :attr:`audiotools.Signal.time` property.

>>> signal = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> print(signal.time)
[0.00000000e+00 2.08333333e-05 4.16666667e-05 ... 9.99937500e-01
 9.99958333e-01 9.99979167e-01]

It's important to understand that all modifications are in-place meaning that calling a method does not return a changed copy of the signal but  directly changes the values of the signal.

>>> signal = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> signal.add_tone(frequency=500)
>>> print(signal.var())
0.49999999999999994

Creating a copy of a Signal requires the explicit use of the :meth:`audiotools.Signal.copy` method. The :meth:`audiotools.Signal.copy_empty` method can be used to create an empty copy with the same shape as the original.

>>> signal = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> signal2 = signal.copy_empty()


Basic signal modifications
==========================

Basic signal modifications such as adding a tone or noise are directly available as methods. Tones are easily added through the :meth:`audiotools.Signal.add_tone` method. A signal with two antiphasic 500Hz tones in the two channels is created by running:

>>> sig = audio.Signal(2, 1, 48000)
>>> sig.ch[0].add_tone(frequency=500, amplitude=1, start_phase=0)
>>> sig.ch[1].add_tone(frequency=500, amplitude=1, start_phase=3.141)

Fade-in and -out ramps with different shapes can be applied using the :meth:`audiotools.Signal.add_fade_window` method.

>>> sig = audio.Signal(1, 1, 48000)
>>> sig.add_tone(frequency=500, amplitude=1, start_phase=0)
>>> sig.add_fade_window(duration=30e-3, type='cos')

Similarly, a cosine modulator cam be added through the :meth:`audiotools.Signal.add_cos_modulator` method.

>>> sig = audio.Signal(1, 1, 48000)
>>> sig.add_cos_modulator(frequency=30, m=1)


Generating Noise
****************

audiotools provides multiple functions to generate noise.

>>> white_noise = audio.Signal(2, 1, 48000).add_noise()
>>> pink_noise = audio.Signal(2, 1, 48000).add_noise(ntype='pink')
>>> brown_noise = audio.Signal(2, 1, 48000).add_noise(ntype='brown')

adds the same white, pink or brown Gaussian noise to all channels of the signal. The noise variance and a seed for the random number generator can be defined by passing the respective argument (see :meth:`audiotools.Signal.add_noise`). Uncorrelated noise can be generated using the :meth:`audiotools.Signal.add_uncorr_noise` method. This uses the Gram-Schmidt process in order to orthoganalize noise tokens in order to minimize variance in the created correlation.

>>> noise = audio.Signal(3, 1, 48000).add_uncorr_noise(corr=0.2, ntype='white')
>>> np.cov(noise.T)
array([[1.00002083, 0.20000417, 0.20000417],
       [0.20000417, 1.00002083, 0.20000417],
       [0.20000417, 0.20000417, 1.00002083]])

There is also an option to create band-limited, partly-correlated or uncorrelated noise by defining low-, high- or band-pass filter that are applied before using the Gram-Schmidt process. For more details please refer to the documentation of
:meth:`audiotools.Signal.add_uncorr_noise`.

Signal statistics
=================

Some basic signal statistics are accessible through the :attr:`audiotools.Signal.stats` subclass. This includes the mean, variance of the channels. All stats are calculated per channel.

>>> noise = audio.Signal(3, 1, 48000).add_noise()
>>> noise.stats.mean
Signal([-2.40525192e-17, -2.40525192e-17, -2.40525192e-17])

>>> noise = audio.Signal(3, 1, 48000).add_noise('pink')
>>> noise.stats.var
Signal([-2.40525192e-17, -2.40525192e-17, -2.40525192e-17])
Signal([1., 1., 1.])

Stats also allows for easy access to the signals full-scale level

>>> noise = audio.Signal(3, 1, 48000).add_noise('pink')
>>> noise.stats.dbfs
Signal([3.01029996, 3.01029996, 3.01029996])

When assuming that the values within the signal represents the sound pressure in pascal, one can also calculate the sound pressure level.

>>> noise = audio.Signal(3, 1, 48000).add_noise('pink')
>>> noise.set_dbspl(70)
>>> noise.stats.dbspl
Signal([93.97940009, 93.97940009, 93.97940009])

Additionally, it is possible to calculate A and C weighted sound pressure level

>>> noise = audio.Signal(3, 1, 48000).add_noise('pink')
>>> noise.stats.dba
Signal([89.10458354, 89.10458354, 89.10458354])

>>> noise = audio.Signal(3, 1, 48000).add_noise('pink')
>>> noise.stats.dbc
Signal([90.82348995, 90.82348995, 90.82348995])

Filtering
=========

audiotools provides access to some often used filters as well as the option to generate filterbanks. Filters can be accessed through the :attr:`audiotools.filter` submodule.

Here, one can either directly call individual filters. The following filters are currently implemented:
  - :func:`audiotools.filter.butterworth`: The butterworth filter implemented by scipy (second order sections)
  - :func:`audiotools.filter.butterworth`: A brickwall filter implemented in the frequency domain
  - :func:`audiotools.filter.gammatone`: A (complex valued) gammatone filter.

When used with the signal class, there is no need to provide a sampling frequency:

>>> sig = audiotools.Signal(2, 1, 48000)
>>> filt_sig = audiotools.filter.gammatone(sig, fc=500, bw=80)

Alternatively, there is also a unified interface for low- high and band-pass filters.
  - :func:`audiotools.filter.lowpass`: Lowpass filter, currently Butterworth or Brickwall
  - :func:`audiotools.filter.highpass`: Highpass filter, currently Butterworth or Brickwall
  - :func:`audiotools.filter.bandpass`: Bandapss filter, currently Butterworth, Brickwall, Gammatone

A third order butterworth filter can thus be implemented as:

>>> sig = audio.Signal(2, 1, 48000)
>>> filt_sig = audio.filter.lowpass(sig, f_cut=1000, filter_type='butter', order=3)

Or:

>>> sig = audio.Signal(2, 1, 48000)
>>> filt_sig = audio.filter.butterworth(sig, low_f=None, high_f=1000, order=3)

The three unified interfaces are also implemented as methods of the :class:`audiotools.Signal` class:

>>> sig = audio.Signal(2, 1, 48000).add_noise()
>>> lp_sig = sig.copy().lowpass(f_cut=1000, filter_type='butter', order=3)
>>> hp_sig = sig.copy().highpass(f_cut=1000, filter_type='butter', order=3)
>>> bp_sig = sig.copy().bandpass(fc=2000, bw=500, filter_type='butter', order=3)

See :meth:`audiotools.Signal.lowpass`, :meth:`audiotools.Signal.highpass` and :meth:`audiotools.Signal.bandpass` for more information.

Filterbanks
************

audiotools provides two commonly used standard banks as well as the option to build custom banks.

Currently the following standard banks are available:
  1. :func:`audiotools.filter.bank.octave_bank` (fractional) Octave filterbank.
  2. :func:`audiotools.filter.bank.auditory_gamma_bank` An auditory gammatone-filterbank.

A 1/3 octave fractional filterbank can be generated as followed:

>>> bank = audio.filter.bank.octave_bank(fs=48000, flow=24.8, fhigh=20158.0, oct_fraction=3)
>>> bank.fc
array([   24.80314144,    31.25      ,    39.37253281,    49.60628287,
          62.5       ,    78.74506562,    99.21256575,   125.        ,
         157.49013124,   198.4251315 ,   250.        ,   314.98026247,
         396.85026299,   500.        ,   629.96052495,   793.70052598,
        1000.        ,  1259.92104989,  1587.40105197,  2000.        ,
        2519.84209979,  3174.80210394,  4000.        ,  5039.68419958,
        6349.60420787,  8000.        , 10079.36839916, 12699.20841575,
       16000.        , 20158.73679832])

With all filter-banks, Signal can either be filtered by applying the whole bank at the same time returning a multi-channel signal

>>> sig = audio.Signal(2, 1, 48000).add_noise()
>>> filt_sig = bank.filt(sig)
>>> filt_sig.n_channels
(2, 30)

Or, alternatively, the filterbank can also be indexed to apply individual filters

>>> filt_sig = bank[2:4].filt(sig)
>>> filt_sig.n_channels
(2, 2)

The :func:`audiotools.filter.bank.create_filterbank` can be used to create custom filterbanks. E.g. a brickwall filterbank with filters around 100Hz, 200Hz and 300Hz with the bandwidths 10Hz, 20Hz and 30Hz can be created as follows:

>>> fc_vec = np.array([100, 200, 300])
>>> bw_vec = np.array([10, 20, 30])
>>> bank = audio.filter.bank.create_filterbank(fc=fc_vec, bw=bw_vec, filter_type='brickwall', fs=48000)
>>> sig = audio.Signal(2, 1, 48000).add_noise()
>>> filt_sig = bank.filt(sig)
>>> filt_sig.n_channels
(2, 3)

Frequency weighting
********************
audiotools implements A and C weighting filters following IEC 61672-1. Both C and A weighted sound pressure levels can be accessed as properties through :attr:`audiotools.Signal.stats`. Additionally, the filters can be applied through :func:`audiotools.filter.a_weighting` and :func:`audiotools.filter.c_weighting`.
