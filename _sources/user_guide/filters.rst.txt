Filtering
=========

The `audiotoolbox` library provides access to commonly used filters as well as the option to generate filterbanks. Filters can be accessed through the :mod:`audiotoolbox.filter` submodule.

Individual Filters
------------------

You can directly call individual filters. The following filters are currently implemented:

- :func:`audiotoolbox.filter.butterworth`: The Butterworth filter implemented by scipy (second order sections).
- :func:`audiotoolbox.filter.brickwall`: A brickwall filter implemented in the frequency domain.
- :func:`audiotoolbox.filter.gammatone`: A (complex valued) gammatone filter.

When used with the `Signal` class, there is no need to provide a sampling frequency:

.. code-block:: python

    import audiotoolbox as audio

    sig = audio.Signal(2, 1, 48000)
    filt_sig = audio.filter.gammatone(sig, fc=500, bw=80)

Unified Interface for Filters
-----------------------------

There is also a unified interface for low-pass, high-pass, and band-pass filters:

- :func:`audiotoolbox.filter.lowpass`: Low-pass filter, currently Butterworth or Brickwall.
- :func:`audiotoolbox.filter.highpass`: High-pass filter, currently Butterworth or Brickwall.
- :func:`audiotoolbox.filter.bandpass`: Band-pass filter, currently Butterworth, Brickwall, Gammatone.

A third-order Butterworth filter can be implemented as follows:

.. code-block:: python

    import audiotoolbox as audio

    sig = audio.Signal(2, 1, 48000)
    filt_sig = audio.filter.lowpass(sig, f_cut=1000, filter_type='butter', order=3)

Or:

.. code-block:: python

    sig = audio.Signal(2, 1, 48000)
    filt_sig = audio.filter.butterworth(sig, low_f=None, high_f=1000, order=3)

The three unified interfaces are also implemented as methods of the :class:`audiotoolbox.Signal` class:

.. code-block:: python

    sig = audio.Signal(2, 1, 48000).add_noise()
    lp_sig = sig.copy().lowpass(f_cut=1000, filter_type='butter', order=3)
    hp_sig = sig.copy().highpass(f_cut=1000, filter_type='butter', order=3)
    bp_sig = sig.copy().bandpass(fc=2000, bw=500, filter_type='butter', order=3)

See :meth:`audiotoolbox.Signal.lowpass`, :meth:`audiotoolbox.Signal.highpass`, and :meth:`audiotoolbox.Signal.bandpass` for more information.

Filterbanks
-----------

`audiotoolbox` provides two commonly used standard banks as well as the option to build custom banks.

Currently, the following standard banks are available:

1. :func:`audiotoolbox.filter.bank.octave_bank`: (fractional) Octave filterbank.
2. :func:`audiotoolbox.filter.bank.auditory_gamma_bank`: An auditory gammatone-filterbank.

A 1/3 octave fractional filterbank can be generated as follows:

.. code-block:: python

    bank = audio.filter.bank.octave_bank(fs=48000, flow=24.8, fhigh=20158.0, oct_fraction=3)
    print(bank.fc)
    # Output: array([   24.80314144,    31.25      ,    39.37253281,    49.60628287,
    #                   62.5       ,    78.74506562,    99.21256575,   125.        ,
    #                  157.49013124,   198.4251315 ,   250.        ,   314.98026247,
    #                  396.85026299,   500.        ,   629.96052495,   793.70052598,
    #                 1000.        ,  1259.92104989,  1587.40105197,  2000.        ,
    #                 2519.84209979,  3174.80210394,  4000.        ,  5039.68419958,
    #                 6349.60420787,  8000.        , 10079.36839916, 12699.20841575,
    #                16000.        , 20158.73679832])

With all filter-banks, a `Signal` can either be filtered by applying the whole bank at the same time, returning a multi-channel signal:

.. code-block:: python

    sig = audio.Signal(2, 1, 48000).add_noise()
    filt_sig = bank.filt(sig)
    print(filt_sig.n_channels)
    # Output: (2, 30)

Or, alternatively, the filterbank can also be indexed to apply individual filters:

.. code-block:: python

    filt_sig = bank[2:4].filt(sig)
    print(filt_sig.n_channels)
    # Output: (2, 2)

The :func:`audiotoolbox.filter.bank.create_filterbank` can be used to create custom filterbanks. For example, a brickwall filterbank with filters around 100Hz, 200Hz, and 300Hz with bandwidths of 10Hz, 20Hz, and 30Hz can be created as follows:

.. code-block:: python

    fc_vec = np.array([100, 200, 300])
    bw_vec = np.array([10, 20, 30])
    bank = audio.filter.bank.create_filterbank(fc=fc_vec, bw=bw_vec, filter_type='brickwall', fs=48000)
    sig = audio.Signal(2, 1, 48000).add_noise()
    filt_sig = bank.filt(sig)
    print(filt_sig.n_channels)
    # Output: (2, 3)

Frequency Weighting
--------------------

`audiotoolbox` implements A and C weighting filters following IEC 61672-1. Both C and A weighted sound pressure levels can be accessed as properties through :attr:`audiotoolbox.Signal.stats`. Additionally, the filters can be applied through :func:`audiotoolbox.filter.a_weighting` and :func:`audiotoolbox.filter.c_weighting`.

.. code-block:: python

    noise = audio.Signal(3, 1, 48000).add_noise('pink')
    print(noise.stats.dba)
    # Output: Signal([89.10458354, 89.10458354, 89.10458354])

    noise = audio.Signal(3, 1, 48000).add_noise('pink')
    print(noise.stats.dbc)
    # Output: Signal([90.82348995, 90.82348995, 90.82348995])
