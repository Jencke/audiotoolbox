Signal Statistics and Levels
============================

Some basic signal statistics are accessible through the :attr:`audiotoolbox.Signal.stats` property. This includes the mean and variance of the channels, calculated per channel. The library also provides convenient methods for level calculations in various units.

Let's create a pink noise signal and explore its properties:

>>> noise = audio.Signal(2, 1, 48000).add_noise('pink')
>>>
>>> # Get basic statistics
>>> print(f"Mean: {noise.stats.mean}")
Mean: Signal([-2.4e-17, -2.4e-17])
>>> print(f"Variance: {noise.stats.var}")
Variance: Signal([1., 1.])
>>>
>>> # Get level in dB Full Scale (dBFS)
>>> print(f"Level in dBFS: {noise.stats.dbfs}")
Level in dBFS: Signal([3.01, 3.01])
>>>
>>> # Get A-weighted and C-weighted levels
>>> print(f"A-weighted SPL: {noise.stats.dba}")
A-weighted SPL: Signal([89.10, 89.10])
>>> print(f"C-weighted SPL: {noise.stats.dbc}")
C-weighted SPL: Signal([90.82, 90.82])

You can also normalize a signal to a target Sound Pressure Level (SPL), assuming the signal values represent pressure in Pascals.

>>> # Normalize the signal to 70 dB SPL
>>> noise.set_dbspl(70)
>>> # The stats.dbspl property will now reflect this level
>>> noise.stats.dbspl
Signal([70., 70.])

Additionally, it is possible to calculate A-weighted and C-weighted sound pressure levels, which are common in acoustic measurements:

There is also the option to get the octave-band levels:

>>> fc, dbfs = noise.stats.octave_band_levels(oct_fraction=1)
>>> print(fc)
[   31.25    62.5    125.     250.     500.    1000.    2000.    4000.
  8000.   16000.  ]
>>> print(dbfs)
[-32.5484477  -31.64357561 -32.14208818 -32.32627542 -32.59523029
 -32.26243379 -32.38507482 -32.36273354 -32.4864307  -32.51044551]
