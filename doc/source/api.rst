.. toctree::
   :maxdepth: 2

API
===

Fluent Interface API
--------------------

.. autoclass:: audiotools.Signal
   :members: fs, n_channels, n_samples, duration, ch, concatenate,
             multiply, add, subtract, abs, time, add_tone, add_noise,
             set_dbspl, calc_dbspl

Function Based API
------------------

.. automodule:: audiotools
    :members: generate_tone, generate_noise, set_dbspl, calc_dbspl
