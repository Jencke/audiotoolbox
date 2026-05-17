Core functions
------------------


.. automodule:: audiotoolbox
    :members:  schroeder_phase, phon_to_dbspl, dbspl_to_phon,
            time2phase, phase2time, crossfade, from_file


Filter Module
^^^^^^^^^^^^^

Individual filters can either be applied by directly calling the respective filter functions such as :func:`filter.gammatone` or :func:`filter.efilt`, or by using the unified interfaces for :func:`filter.bandpass`, :func:`filter.lowpass` and :func:`filter.highpass` filters. When using the unified interface, all additional arguments are passed to the respective filter functions.

.. automodule:: audiotoolbox.filter
    :members: bandpass, lowpass, highpass, gammatone, butterworth, brickwall, efilt, a_weighting, c_weighting

Filterbanks are created using the :func:`create_filterbank` command

.. automodule:: audiotoolbox.filter.bank
    :members: create_filterbank, auditory_gamma_bank, octave_bank

