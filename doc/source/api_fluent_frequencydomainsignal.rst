Signals in the frequency domain (audiotools.FrequencyDomainSignal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `FrequencyDomainSignal` class also inherits from `numpy.ndarray` via the
`audiotools.BaseSignal` class:

.. inheritance-diagram:: audiotools.FrequencyDomainSignal

.. autoclass:: audiotools.FrequencyDomainSignal
   :members: fs, n_channels, n_samples, duration, ch, concatenate,
             multiply, add, abs, freq, phase, mag, time_shift,
             phase_shift, to_timedomain

.. _docs: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
