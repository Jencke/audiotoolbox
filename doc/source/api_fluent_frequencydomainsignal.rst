Signals in the frequency domain (audiotoolbox.FrequencyDomainSignal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `FrequencyDomainSignal` class also inherits from `numpy.ndarray` via the
`audiotoolbox.BaseSignal` class:

.. inheritance-diagram:: audiotoolbox.FrequencyDomainSignal

.. autoclass:: audiotoolbox.FrequencyDomainSignal
   :members: fs, n_channels, n_samples, duration, ch, concatenate,
             multiply, add, abs, freq, phase, mag, time_shift,
             phase_shift, to_timedomain

.. _docs: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
