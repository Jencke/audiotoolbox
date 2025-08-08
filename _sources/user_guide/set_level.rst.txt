.. _set_level:

Determining and Setting Levels
==============================

This section provides an overview of how to determine and set signal levels
using the :class:`~audiotoolbox.Signal` class and its
:attr:`~audiotoolbox.Signal.stats` property.

Getting Signal Statistics
-------------------------

All level calculations and statistics are accessed through the ``.stats``
property, which returns a :class:`~audiotoolbox.SignalStats` object.
This provides convenient access to common metrics, calculated per channel.

Let's create a noise signal to demonstrate:

.. code-block:: python

   import audiotoolbox as audio

   # Create a two-channel noise signal
   sig = audio.Signal(n_channels=2, duration=1, fs=48000).add_noise()

The following properties are available:

* **.stats.rms**: The Root-Mean-Square level of the signal.
* **.stats.dbspl**: The level in dB Sound Pressure Level (SPL), assuming
    the signal values are pressure in Pascals relative to 20 µPa.
* **.stats.dbfs**: The level in dB Full Scale, where 0 dBFS is a sine
    wave with an amplitude of 1.
* **.stats.dba** and **.stats.dbc**: A- and C-weighted SPL.
* **.stats.crest_factor**: The ratio of the peak amplitude to the RMS value.

.. code-block:: python

   # Get various level and statistical properties
   rms_val = sig.stats.rms
   spl_val = sig.stats.dbspl
   dbfs_val = sig.stats.dbfs
   crest_val = sig.stats.crest_factor

   print(f"RMS: {rms_val}")
   print(f"SPL: {spl_val:.2f} dB")
   print(f"dBFS: {dbfs_val:.2f} dB")
   print(f"Crest Factor: {crest_val:.2f} dB")


Setting and Normalizing Levels
------------------------------

To change a signal's level, use the methods directly available on the
``Signal`` object.

Setting Sound Pressure Level (SPL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotoolbox.Signal.set_dbspl` method normalizes the signal
to a target SPL.

.. code-block:: python

   # Normalize the signal to 70 dB SPL
   sig.set_dbspl(70)

   # The .stats.dbspl property will now reflect this new level
   print(f"New SPL: {sig.stats.dbspl:.2f} dB")

Setting dBFS
~~~~~~~~~~~~

Similarly, :meth:`~audiotoolbox.Signal.set_dbfs` normalizes the signal to
a target dBFS value.

.. code-block:: python

   # Normalize the signal to -6 dBFS
   sig.set_dbfs(-6)

   print(f"New dBFS: {sig.stats.dbfs:.2f} dB")

Relative Level Adjustments
~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods can be used to set levels relatively. For example, to set
one signal to a 10 dB higher level than another:

.. code-block:: python

   # Create two signals
   sig1 = audio.Signal(n_channels=2, duration=1, fs=48000).add_noise()
   sig2 = audio.Signal(n_channels=2, duration=1, fs=48000).add_noise()

   # Set the level of sig1 to be 10 dB higher than sig2
   sig1.set_dbfs(sig2.stats.dbfs + 10)

You can also apply this to individual channels. To set the first channel
to a 5 dB lower level than the second channel:

.. code-block:: python

   # Set channel 0 to be 5 dB lower than channel 1
   sig.ch[0].set_dbfs(sig.ch[1].stats.dbfs - 5)
