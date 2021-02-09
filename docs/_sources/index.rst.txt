.. audiotools documentation master file, created by
   sphinx-quickstart on Thu Jan 14 08:08:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   api



.. image:: logo.png
	   :width: 7cm
           :align: center

====================
What is audiotools ?
====================
**audiotools** is a python package designed to generate and analyze
acoustic stimuli for use in auditory research. It aims to provide an
easy to use and intuitive interface.

**auditools** provides the powerfull `Signal` class which extends the
standard `numpy` array class with a fluent interface that provides
methods and attributes often used in auditory signal processing.

The commands:

>>> import audiotools as audio
>>> sig = audio.Signal(n_channels=1, duration=1, fs=48000)
>>> sig.add_tone(500).set_dbspl(60).add_fade_window(10e-3, 'cos')

create a 1 second long signal with 1 channel at a sampling rate of
48kHz. A 500 Hz tone is then added to this signal, the level is set to
60dB SPL and a 10ms raised cosine fade-in and fade-out is added. It
also provides method to quickly switch between the frequency and
time-domain representation of the same signal:

>>> sig.add_noise()
>>> f_sig = sig.to_freqdomain()
>>> f_sig[f_sig.freq.abs() > 1000] = 0
>>> sig = f_sig.to_timedomain()

first adds white noise to the signal and then sets all spectral
components above 1kHz to zero.

As all Signal Classes are extensions of the `numpy` array, they can be
used as drop-in replacements.

Installation
============

Currently, the easiest method is to install
audiotools by cloning the git repository:

 1. Clone the repository: `git clone https://github.com/Jencke/audiotools.git`
 2. Install the package: `pip install ./`
 3. Optionally run the tests: `pytest`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
