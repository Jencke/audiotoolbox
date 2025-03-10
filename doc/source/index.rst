
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   user_guide
   api



.. image:: logo.png
	   :width: 7cm
           :align: center

====================
What is audiotoolbox ?
====================
**audiotoolbox** is a python package designed to generate and analyze acoustic stimuli for use in auditory research. It aims to provide an easy to use and intuitive interface.

**auditools** provides the powerfull `Signal` class which extends the standard `numpy` array class with a fluent interface that provides methods and attributes often used in auditory signal processing.

The commands:

>>> import audiotoolbox as audio
>>> sig = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> sig.add_tone(500).set_dbspl(60).add_fade_window(10e-3, 'cos')

create a 1 second long signal with 1 channel at a sampling rate of 48kHz. A 500 Hz tone is then added to this signal, the level is set to 60dB SPL and a 10ms raised cosine fade-in and fade-out is added.

The Signal class also provides method to quickly switch between the frequency and time-domain representation of the same signal:

>>> sig.add_noise()
>>> f_sig = sig.to_freqdomain()
>>> f_sig[f_sig.freq.abs() > 1000] = 0
>>> sig = f_sig.to_timedomain()

first adds gaussian white noise to the signal and then sets all spectral components above 1kHz to zero.

All Signal classes are extensions of the standard `numpy` array, they can be used as drop-in replacements. As a consequence, the Signal class also inherits all methods of `numpy.ndarray`:

>>> import numpy as np
>>> sig = audio.Signal(n_channels=3, duration=1, fs=48000)
>>> sig.add_uncorr_noise(0.5)
>>> sig.var(axis=0)
Signal([1., 1., 1.])

More information and a detailed documentation of the methods and functions provided by audiotoolbox can be found in the `api` and `introduction` sections.

Installation
============

Using pip
----------

You can use pip to install audiotoolbox

.. code-block:: bash

    pip install audiotoolbox

From GitHub
-----------

Or directly from GitHub

 1. Clone the repository: `git clone https://github.com/Jencke/audiotoolbox.git`
 2. Install the package: `pip install ./`
 3. Optionally run the tests: `pytest`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
