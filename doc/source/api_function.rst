Function Based API
------------------

The function based interface provides most of the functions that are
availible as methods of the :func:`Signal` and also some that are not
directly availible through the :func:`Signal` class.


.. automodule:: audiotools
    :members: crest_factor, shift_signal,
              calc_coherence, schroeder_phase, phon_to_dbspl,
              time2phase, phase2time, nsamples, lowpass, crossfade


audiotools.filter
^^^^^^^^^^^^^^^^^

Individual filter can either be applied by directly calling the respective filter functions such as :func:`filter.gammatone` or by using the unified interfaces for :func:`filter.bandpass`, :func:`filter.lowpass` and :func:`filter.highpass` filters. When using the unified interface, all additional arguments are passed to the respective filter functions.

.. automodule:: audiotools.filter
    :members: bandpass, lowpass, highpass, gammatone, butterworth, brickwall

Filterbanks are created using the :func:`create_filterbank` command

.. automodule:: audiotools.filter
    :members: create_filterbank
