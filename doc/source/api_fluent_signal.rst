Signals in the time domain (audiotools.Signal)
^^^^^^^^^^^^^^^^^

The `Signal` Class inherits from `numpy.ndarray` via the
`audiotools.BaseSignal` class:

.. inheritance-diagram:: audiotools.Signal

As a consequence, `numpy.ndarray` methods such as `x.min()`,
`x.max()`, `x.sum()`, `x.var()` and others can also be used on
auditools.Signal objects. For more informations check the numpy docs_.

.. autoclass:: audiotools.Signal
   :members: fs, n_channels, n_samples, duration, ch, concatenate, multiply,
             add, abs, time, add_tone, add_noise, set_dbspl, set_dbfs, bandpass,
             zeropad, add_fade_window, add_cos_modulator, delay, phase_shift,
             clip, rms, rectify, to_freqdomain, add_uncorr_noise

.. _docs: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
