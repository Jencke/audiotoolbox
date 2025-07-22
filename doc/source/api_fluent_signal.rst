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
            add, abs, time, add_tone, add_noise, set_dbspl, set_dbfs, bandpass,
            zeropad, add_fade_window, add_cos_modulator, delay, phase_shift,
            clip, rectify, to_freqdomain, add_uncorr_noise, from_file, write_file

The `Signal.stats` attribute
============================

The `Signal.stats` attribute gives access to an instance of the `audiotoolbox.oaudio.stats.SignalStats` class.
e.g.:

>>> sig = audio.Signal(1, 1, 48000)
>>> rms = sig.stats.rms

.. autoclass:: audiotoolbox.oaudio.stats.SignalStats
   :members: rms, mean, var, dbspl, dbfs, crest_factor, dba, dbc, octave_band_levels



.. _docs: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
