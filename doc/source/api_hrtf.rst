Head-related impulse responses (audiotoolbox.HRIRSet)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `HRIRSet` class holds a set of head-related impulse responses (HRIRs)
measured over a number of spatial directions. The impulse responses are
stored as a regular :class:`audiotoolbox.Signal` of shape
``(n_taps, n_directions, 2)`` while the source directions are kept in a
separate position table. It provides direction lookup, interpolation, and
binaural rendering, and can load measurements from SOFA files.

.. autoclass:: audiotoolbox.HRIRSet
   :members: from_sofa, fs, n_directions, n_taps, azimuth, elevation,
            distance, to_hrtf, nearest, interpolate, render, summary
