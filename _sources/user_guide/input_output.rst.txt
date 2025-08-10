.. _signal_io:

Loading and saving audio files
====================

This section explains how to load and save signals using the `audiotoolbox` library. 
The `Signal` class provides methods for reading signals from audio files and writing signals to audio files. 
The library supports all audio file formats supported by libsndfile, such as WAV, FLAC, AIFF, and more.

Loading Signals
---------------

To load a signal from an audio file, you can use the `from_file` method of the `Signal` class. This method reads a signal from an audio file and returns it as a `Signal` object. You can specify the start point and the channels to load.

Example
~~~~~~~

A signal from a file can either be loaded into a new `Signal` object or an existing one. 

To load a signal into a new `Signal` object, you can use the following code:

.. code-block:: python

    from audiotoolbox import Signal

    # Load the signal from the file "example.wav" into a new Signal object
    sig = Signal.from_file("example.wav")

This code creates a new `Signal` object and loads the signal from the file "example.wav" into it. The sample rate and number of channels are automatically determined from the file.

If you want to load the signal into an existing `Signal` object, you can use the following code:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig = Signal(2, 1, 48000)

    # Load the signal from the file "example.wav"
    sig.from_file("example.wav")

In this case, the signal is loaded into the existing `Signal` object `sig`. The sample rate and number of channels of the file must match the `Signal` object.
If you want to load only a portion of the signal or a specific channel, you can specify additional parameters:

- `start`: The starting sample index to load from the file.
- `channels`: The channel index to load from the file.

To read only a portion of the signal starting at sample index 1000, you can use the following code:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 1 channel, 1 second duration, and 48 kHz sampling rate
    sig = Signal(1, 1, 48000)

    # Load the signal from the file "example.wav" starting at sample index 1000
    sig.from_file("example.wav", start=1000, channels=0)

Saving Signals
--------------

To save a signal to an audio file, you can use the `write_file` method of the `Signal` class. This method saves the current signal as an audio file. You can specify additional parameters for the file format through keyword arguments.

Example
~~~~~~~

Save the signal to a file named "output.wav":

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig = Signal(2, 1, 48000)

    # Save the signal to the file "output.wav"
    sig.write_file("output.wav")

Save the signal to a file with a specific format and subtype:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig = Signal(2, 1, 48000)

    # Save the signal to the file "output.wav" with format "WAV" and subtype "PCM_16"
    sig.write_file("output.wav", format="WAV", subtype="PCM_16")

Save the signal to a FLAC file:

.. code-block:: python

    from audiotoolbox import Signal

    # Create a Signal object with 2 channels, 1 second duration, and 48 kHz sampling rate
    sig = Signal(2, 1, 48000)

    # Save the signal to the file "output.flac" with format "FLAC"
    sig.write_file("output.flac", format="FLAC")

See Also
--------

- :func:`audiotoolbox.from_file` : Function to read an audio file into a new Signal object.
- :meth:`audiotoolbox.Signal.from_file` : Method to load a signal into an existing Signal object.
- :meth:`audiotoolbox.Signal.write_file` : Method to save a signal to an audio file.
