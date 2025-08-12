.. _guide_playback:

Playback
========
The :meth:`audiotoolbox.Signal.play` method can be used to quickly listen to the signal using the default device.

>>> sig = audio.Signal(1, 1, 48000).add_tone(500).add_fade_window(30e-3)
>>> sig.play()