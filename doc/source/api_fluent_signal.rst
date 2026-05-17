Signals in the time domain (audiotoolbox.Signal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Signal` Class inherits from `numpy.ndarray` via the
`audiotoolbox.BaseSignal` class:

.. inheritance-diagram:: audiotoolbox.Signal

As a consequence, `numpy.ndarray` methods such as `x.min()`,
`x.max()`, `x.sum()`, `x.var()` and others can also be used on
auditools.Signal objects. For more informations check the numpy docs_.

.. autoclass:: audiotoolbox.Signal
   :members: fs, n_channels, n_samples, duration, ch, concatenate, multiply,
            add, abs, time, add_tone, add_noise, set_dbspl, set_dbfs, lowpass,
            highpass, bandpass,
            zeropad, add_fade_window, add_cos_modulator, delay, phase_shift,
            trim, rectify, to_freqdomain, add_uncorr_noise, from_file, write_file,
            convolve, play, resample

The `Signal.stats` sub-module
=============================

The `Signal.stats` submodule gives access to an instance of the `audiotoolbox.stats.SignalStats` class.
e.g.:

>>> sig = audio.Signal(1, 1, 48000)
>>> rms = sig.stats.rms

.. autoclass:: audiotoolbox.stats.SignalStats
   :members: rms, mean, var, dbspl, dbfs, crest_factor, dba, dbc, octave_band_levels

The `Signal.time_frequency` sub-module
======================================

The `Signal.time_frequency` submodule gives access to an instance of the `audiotoolbox.time_frequency.TimeFrequency` class that provides time-frequency analysis methods such as spectrograms.

.. autoclass:: audiotoolbox.time_frequency.TimeFrequency
   :members: octave_band_specgram, gammatone_specgram, filterbank_specgram, stft_specgram

The `Signal.viz` sub-module
===========================

The `Signal.viz` submodule gives access to an instance of the `audiotoolbox.viz.Visualization` class that provides plotting and visualization helpers.

.. autoclass:: audiotoolbox.viz.Visualization
   :members: plot, spectrum, specgram_overview

.. _docs: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
