.. toctree::
   :maxdepth: 2

API
===
Fluent Interface API
--------------------

The Signal Class inherits from numpy.ndarray via the BaseSignal class:

.. inheritance-diagram:: audiotools.Signal

As a consequence, `numpy.ndarray` methods such as `x.min()`,
`x.max()`, `x.sum()`, `x.var()` and others can also be used on
auditools.Signal objects. For more informations check the numpy docs_.

.. autoclass:: audiotools.Signal
   :members: fs, n_channels, n_samples, duration, ch, concatenate,
             multiply, add, abs, time, add_tone, add_noise,
             set_dbspl, calc_dbspl, set_dbfs, calc_dbfs, calc_crest_factor,
	     bandpass, zeropad, add_fade_window, add_cos_modulator, delay,
	     phase_shift, clip, rms, rectify, to_freqdomain, add_uncorr_noise

.. autoclass:: audiotools.FrequencyDomainSignal
   :members: fs, n_channels, n_samples, duration, ch, concatenate,
             multiply, add, abs, freq, phase, mag, time_shift,
             phase_shift, to_timedomain

Function Based API
------------------

.. automodule:: audiotools
    :members: generate_tone, generate_noise, set_dbspl, calc_dbspl,
              set_dbfs, calc_dbfs, crest_factor, zeropad,
              gaussian_fade_window, cosine_fade_window,
              cos_amp_modulator, shift_signal, generate_uncorr_noise,
              calc_coherence, schroeder_phase, phon_to_dbspl,
              time2phase, phase2time, nsamples


Filter
------

.. automodule:: audiotools.filter
    :members: gammatone, brickwall

.. _docs: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
