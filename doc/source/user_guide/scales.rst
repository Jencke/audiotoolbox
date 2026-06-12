.. _guide_scales:

Frequency Scales
================

`audiotoolbox` provides conversions between frequency (Hz) and commonly used
psychoacoustic scales:

- Bark
- ERB (Equivalent Rectangular Bandwidth rate)
- Fractional octave-band number
- Mel
- Semitone (MIDI-like)
- Greenwood place-frequency mapping

The scales API is available through ready-to-use instances:

>>> import audiotoolbox as audio
>>> audio.bark
<audiotoolbox.scales.bark.BarkScale object at ...>
>>> audio.erb
<audiotoolbox.scales.erb.ErbScale object at ...>
>>> audio.octave
<audiotoolbox.scales.octave.OctaveScale object at ...>

All scale objects expose the same core methods:

- ``from_freq(...)``: convert frequency in Hz to scale value
- ``to_freq(...)``: convert from scale value to frequency in Hz
- ``get_bw(...)``: bandwidth in Hz for a center frequency
- ``calc_bw(...)``: compatibility alias for ``get_bw(...)``

Bark Scale
----------

Use Bark when working with critical-band representations.

>>> import numpy as np
>>> freqs = np.array([125, 500, 1000, 4000])
>>> b = audio.bark.from_freq(freqs)
>>> b
array([ 1.21571703,  4.91918699,  8.52743243, 17.46328859])
>>> audio.bark.to_freq(b)
array([ 125.,  500., 1000., 4000.])

Bark bandwidth (critical bandwidth approximation):

>>> audio.bark.get_bw(1000)
162.21671568515828

The tabulated Bark limits can be retrieved with:

>>> audio.bark.get_bark_limits()
[20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]

Note that Bark conversion is defined for approximately ``20 <= f < 15500`` Hz.
Out-of-range values raise ``ValueError``.

ERB Scale
---------

Use ERB values for auditory-filter spacing and filterbank design.

>>> freqs = np.array([125, 500, 1000, 4000])
>>> e = audio.erb.from_freq(freqs)
>>> e
array([ 4.03776804, 10.73247194, 15.57201668, 27.02164247])
>>> audio.erb.to_freq(e)
array([ 125.,  500., 1000., 4000.])

ERB bandwidth at a center frequency:

>>> audio.erb.get_bw(1000)
132.639

Fractional Octave Scale
-----------------------

The octave scale converts between frequency and octave-band numbers.
By convention, the 1000 Hz band is centered around band number 30.

>>> audio.octave.from_freq(1000)
30.0
>>> audio.octave.to_freq(30)
1000.0

For fractional-octave systems, set ``oct_fraction``:

>>> bands = audio.octave.from_freq([500, 1000, 2000], oct_fraction=3)
>>> bands
array([27., 30., 33.])
>>> audio.octave.to_freq(bands, oct_fraction=3)
array([ 500., 1000., 2000.])

Supported base systems are ``base_system=2`` and ``base_system=10``.

Bandwidth for a fractional-octave band:

>>> audio.octave.get_bw(1000, oct_fraction=3, base_system=2)
231.56333016903375

``oct_fraction`` must be a positive integer; invalid values raise
``ValueError``.

Mel Scale
---------

The Mel scale is commonly used in speech/audio features and provides a
perceptual mapping from frequency to a quasi-linear low-frequency and
logarithmic high-frequency scale.

.. code-block:: python

	import audiotoolbox as audio

	mel_vals = audio.mel.from_freq([125, 500, 1000, 4000])
	freqs = audio.mel.to_freq(mel_vals)
	bw = audio.mel.get_bw(1000)

Semitone Scale
--------------

The semitone scale maps frequency to a MIDI-like continuous note number.
Default reference is A4 = 440 Hz at note 69.

.. code-block:: python

	import audiotoolbox as audio

	notes = audio.semitone.from_freq([220, 440, 880])
	freqs = audio.semitone.to_freq(notes)
	bw = audio.semitone.get_bw(1000)

Greenwood Scale
---------------

The Greenwood scale maps frequency to cochlear place using the Greenwood
equation (human defaults are used by default).

.. code-block:: python

	import audiotoolbox as audio

	x = audio.greenwood.from_freq([125, 500, 1000, 4000])
	freqs = audio.greenwood.to_freq(x)
	bw = audio.greenwood.get_bw(1000)

Deprecated Core Wrapper Functions
---------------------------------

For backward compatibility, `audiotoolbox` still exposes legacy helper
functions in the top-level namespace. These wrappers now emit
``DeprecationWarning`` and should be replaced with the scale-object API.

Deprecated wrappers:

- ``audio.get_bark_limits(...)``
- ``audio.freq_to_bark(...)`` and ``audio.bark_to_freq(...)``
- ``audio.freq_to_erb(...)`` and ``audio.erb_to_freq(...)``
- ``audio.freq_to_octband(...)`` and ``audio.octband_to_freq(...)``
- ``audio.calc_bandwidth(...)``

Recommended replacements:

- ``audio.get_bark_limits()`` -> ``audio.bark.get_bark_limits()``
- ``audio.freq_to_bark(f)`` -> ``audio.bark.from_freq(f)``
- ``audio.bark_to_freq(b)`` -> ``audio.bark.to_freq(b)``
- ``audio.freq_to_erb(f)`` -> ``audio.erb.from_freq(f)``
- ``audio.erb_to_freq(e)`` -> ``audio.erb.to_freq(e)``
- ``audio.freq_to_octband(f, ...)`` -> ``audio.octave.from_freq(f, ...)``
- ``audio.octband_to_freq(n, ...)`` -> ``audio.octave.to_freq(n, ...)``
- ``audio.calc_bandwidth(fc, 'cbw')`` -> ``audio.bark.get_bw(fc)``
- ``audio.calc_bandwidth(fc, 'erb')`` -> ``audio.erb.get_bw(fc)``
- ``audio.calc_bandwidth(fc, 'oct')`` -> ``audio.octave.get_bw(fc)``

Frequency Grids on Perceptual Scales
------------------------------------

The core module also provides helpers to create frequency vectors with
uniform spacing on Bark/ERB (and octave stepping):

>>> audio.freqspace(100, 12000, 24, scale='bark')
array([  100.        ,   195.48602755,   285.31169734,   381.406472  ,
	  486.09433527,   600.58186549,   726.31262467,   865.02834524,
	 1018.85010641,  1190.38755782,  1382.88818708,  1600.4448737 ,
	 1848.29013092,  2133.22243676,  2464.23942826,  2853.50536456,
	 3317.87859436,  3881.41758407,  4579.68432764,  5467.55330545,
	 6617.72959366,  7882.67567824,  9585.23586555, 12000.        ])
>>> audio.freqarange(100, 12000, 1, scale='erb')
array([  100.        ,   137.48031099,   179.23261997,   225.74384839,
	  277.55641688,   335.27457107,   399.57142839,   471.19682804,
	  550.98607574,   639.86968518,   738.8842298 ,   849.18443149,
	  972.05662706,  1108.93376974,  1261.41214048,  1431.26996398,
	 1620.48814662,  1831.27337809,  2066.08386609,  2327.65800433,
	 2619.04630806,  2943.64698962,  3305.24558887,  3708.05912073,
	 4156.78525457,  4656.65709917,  5213.50423191,  5833.82068421,
	 6524.84067582,  7294.62298133,  8152.1449128 ,  9107.40701439,
	10171.54969025, 11356.98312541])
>>> audio.freqarange(16, 16000, 1/3, scale='octave')
array([   19.6862664 ,    24.80314144,    31.25      ,    39.37253281,
	   49.60628287,    62.5       ,    78.74506562,    99.21256575,
	  125.        ,   157.49013124,   198.4251315 ,   250.        ,
	  314.98026247,   396.85026299,   500.        ,   629.96052495,
	  793.70052598,  1000.        ,  1259.92104989,  1587.40105197,
	 2000.        ,  2519.84209979,  3174.80210394,  4000.        ,
	 5039.68419958,  6349.60420787,  8000.        , 10079.36839916,
	12699.20841575])
