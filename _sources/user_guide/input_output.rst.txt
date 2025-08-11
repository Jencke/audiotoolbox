.. _signal_io:

Loading and Saving Audio Files
==============================

This section explains how to load and save signals using **audiotoolbox**.
The :class:`~audiotoolbox.Signal` class provides methods for reading from
and writing to audio files. The library supports all audio file formats
backed by `libsndfile <http://www.mega-nerd.com/libsndfile/>`_, such as
WAV, FLAC, and AIFF.

Loading Audio Files
-------------------

There are two primary ways to load an audio file: creating a new ``Signal``
object directly from a file, or loading audio data into an existing ``Signal``.

Creating a Signal from a File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most direct way to load an audio file is to use the
:meth:`~audiotoolbox.Signal.from_file` class method. This creates a new
``Signal`` object with the properties (channel count, sample rate)
inferred from the file.

.. code-block:: python

   import audiotoolbox as audio

   # Load the signal from "example.wav" into a new Signal object
   sig = audio.Signal.from_file("example.wav")

Loading Data into an Existing Signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also load audio data into a ``Signal`` object that you have already
created. When doing this, the sample rate and number of channels of the
file must match the existing ``Signal`` object.

.. code-block:: python

   import audiotoolbox as audio

   # Create a Signal object
   sig = audio.Signal(n_channels=2, duration=1, fs=48000)

   # Load the signal from "example.wav" into the existing object
   sig.from_file("example.wav")

Reading a Portion of a File
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``from_file`` method allows you to load only a specific portion of the
audio file by using the ``start`` and ``channels`` parameters.

.. code-block:: python

   import audiotoolbox as audio

   # Create a Signal object to hold the partial data
   sig = audio.Signal(n_channels=1, duration=1, fs=48000)

   # Load only the first channel from "example.wav", starting at sample 1000
   sig.from_file("example.wav", start=1000, channels=0)

Saving Audio Files
------------------

To save a signal, use the :meth:`~audiotoolbox.Signal.write_file` method.
The file format is typically inferred from the file extension (e.g., ``.wav``,
``.flac``), but can be specified explicitly.

Save a signal to a standard WAV file:

.. code-block:: python

   import audiotoolbox as audio

   # Create a signal
   sig = audio.Signal(n_channels=2, duration=1, fs=48000).add_noise()

   # Save the signal to "output.wav"
   sig.write_file("output.wav")

You can pass keyword arguments to control the output format and subtype.
For example, to save as a 16-bit PCM WAV file:

.. code-block:: python

   # Save the signal as a 16-bit PCM WAV file
   sig.write_file("output.wav", format="WAV", subtype="PCM_16")

Saving to other formats like FLAC is just as easy:

.. code-block:: python

   # Save the signal to a FLAC file
   sig.write_file("output.flac")