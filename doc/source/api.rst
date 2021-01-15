.. toctree::
   :maxdepth: 2

API
===
Fluent Interface API
--------------------

.. autoclass:: audiotools.Signal
   :members: fs, n_channels, n_samples, duration, ch, concatenate,
             multiply, add, abs, time, add_tone, add_noise,
             set_dbspl, calc_dbspl, set_dbfs, calc_dbfs, calc_crest_factor,
	     bandpass

Function Based API
------------------

.. automodule:: audiotools
    :members: generate_tone, generate_noise, set_dbspl, calc_dbspl,
              set_dbfs, calc_dbfs, crest_factor

Filter
------

.. automodule:: audiotools.filter
    :members: gammatone, brickwall
