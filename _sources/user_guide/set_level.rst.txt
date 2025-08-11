.. _set_level:

Determining and Setting Levels
==============================

This section provides an overview and introduction on how to determine and set levels using the `Signal` class and the `SignalStats` class in the `audiotoolbox` library. The `Signal` class provides methods for calculating the root mean square (RMS) value, setting the sound pressure level (SPL), and normalizing the signal to a given dBFS RMS value. 
The `SignalStats` sub_class provides additional methods for calculating various signal statistics.

Calculating RMS
---------------

The RMS value of a signal is a measure of its average power. The `rms` method of the `Signal` class calculates the RMS value of the signal.

Example
~~~~~~~

Calculate the RMS value of a signal:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig = Signal(2, 1, 48000)

    # Calculate the RMS value of the signal
    rms_value = sig.rms()
    print(f"RMS value: {rms_value}")

Setting Sound Pressure Level (SPL)
----------------------------------

The SPL of a signal is a measure of its loudness. The `set_dbspl` method of the `Signal` class normalizes the signal to a given SPL in dB relative to 20e-6 Pa.

Example
~~~~~~~

Set the SPL of a signal to 70 dB:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig = Signal(2, 1, 48000)

    # Set the SPL of the signal to 70 dB
    sig.set_dbspl(70)

Setting dBFS RMS Value
----------------------

The dBFS RMS value of a signal is a measure of its amplitude relative to the full scale. 
The `set_dbfs` method of the `Signal` class normalizes the signal to a given dBFS RMS value.

Example
~~~~~~~

Set the dBFS RMS value of a signal to -3 dB:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig = Signal(2, 1, 48000)

    # Set the dBFS RMS value of the signal to -3 dB
    sig.set_dbfs(-3)

Example
~~~~~~~

Set the level of one signal 10db above another signal:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig1 = Signal(2, 1, 48000)
    sig2 = Signal(2, 1, 48000)

    # Set the level of sig1 10db above sig2
    sig1.set_dbfs(sig2.stats.dbfs + 10)

Set the level of channel one of a signal 5 db below channel two:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig = Signal(2, 1, 48000)

    # Set the level of channel one 5 db below channel two
    sig.ch(1).set_dbfs(sig.ch(2).stats.dbfs - 5)


Calculating Signal Statistics
-----------------------------

The `SignalStats` class provides methods for calculating various signal statistics, such as 
SPL, dBFS, crest factor, and A-weighted and C-weighted SPL.

Example
~~~~~~~

Calculate the SPL, dBFS, and crest factor of a signal:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig = Signal(2, 1, 48000)

    # Calculate the SPL of the signal
    spl_value = sig.stats.dbspl
    print(f"SPL value: {spl_value} dB")

    # Calculate the dBFS of the signal
    dbfs_value = sig.stats.dbfs
    print(f"dBFS value: {dbfs_value} dB")

    # Calculate the crest factor of the signal
    crest_factor_value = sig.stats.crest_factor
    print(f"Crest factor: {crest_factor_value} dB")

    # Calculate the rms value of the signal
    rms_value = sig.stats.rms
    print(f"RMS value: {rms_value}")

Calculate the A-weighted and C-weighted SPL of a signal:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig = Signal(2, 1, 48000)

    # Calculate the A-weighted SPL of the signal
    dba_value = sig.stats.dba
    print(f"A-weighted SPL: {dba_value} dB")

    # Calculate the C-weighted SPL of the signal
    dbc_value = sig.stats.dbc
    print(f"C-weighted SPL: {dbc_value} dB")


See Also
--------

- :meth:`audiotoolbox.Signal.rms` : Method to calculate the RMS value of the signal.
- :meth:`audiotoolbox.Signal.set_dbspl` : Method to set the SPL of the signal.
- :meth:`audiotoolbox.Signal.set_dbfs` : Method to set the dBFS RMS value of the signal.
- :meth:`audiotoolbox.SignalStats.dbspl` : Property to calculate the SPL of the signal.
- :meth:`audiotoolbox.SignalStats.dbfs` : Property to calculate the dBFS of the signal.
- :meth:`audiotoolbox.SignalStats.crest_factor` : Property to calculate the crest factor of the signal.
- :meth:`audiotoolbox.SignalStats.dba` : Property to calculate the A-weighted SPL of the signal.
- :meth:`audiotoolbox.SignalStats.dbc` : Property to calculate the C-weighted SPL of the signal.