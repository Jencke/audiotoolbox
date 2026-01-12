Stimulus visualization
======================

The visualization submodule provides methods for visualizing stimuli.

Timedomain visualization
------------------------

The viz.plot method allowes for a quick visualization of the timedomain signal.

.. plot::
    :include-source:

    sig = audio.Signal(2, 100e-3, 48000)
    sig.ch[0].add_tone(100)
    sig.ch[1].add_tone(100, start_phase=np.pi)
    sig.add_fade_window(20e-3)
    fig, ax = sig.viz.plot()
    fig.show()




Signal Overview
---------------

The specgram_overview method shows an overview that contains a plot of the timedomain-signal, the 1/3rd octave band spectrogram, and the 1/3 octave band levels. As well as some signal statistics.

.. plot::
    :include-source:

    sig = (
    audio.Signal(1, 1, 48000)
    .add_noise("pink")
    .bandpass(500, 500, "butter", order=2)
    .add_cos_modulator(6, 1)
    .set_dbfs(-10)
    )
    fig, ax = sig.viz.specgram_overview()
    fig.show()