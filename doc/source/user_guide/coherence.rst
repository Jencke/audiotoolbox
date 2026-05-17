.. _guide_coherence:

Coherence and Correlation
=========================

`audiotoolbox` includes functions for complex-valued correlation analysis,
useful for binaural and narrowband signal analysis.

Complex Correlation Coefficient
-------------------------------

Use ``cmplx_corr`` to compute the complex correlation coefficient between two
signals. The magnitude describes similarity, and the phase reflects relative
phase differences.

.. code-block:: python

   import audiotoolbox as audio
   import numpy as np

   sig = audio.Signal(n_channels=2, duration=1, fs=48000)
   sig.ch[0].add_tone(500)
   sig.ch[1].add_tone(500, start_phase=np.pi / 4)

   corr = audio.cmplx_corr(sig.ch[0], sig.ch[1])
   print(corr)

Complex Cross-Correlation Over Time
-----------------------------------

For time-varying analysis, use ``cmplx_crosscorr`` with a sliding window.
This is useful when phase or level relations change over time.

.. code-block:: python

   import audiotoolbox as audio

   sig = audio.Signal(n_channels=2, duration=2, fs=48000).add_noise('white')

   # Example: 20 ms analysis window
   cc = audio.cmplx_crosscorr(sig.ch[0], sig.ch[1], win_len=20e-3)
   print(cc.shape)

Binaural Difference Extraction
------------------------------

For narrowband signals, ``extract_binaural_differences`` provides
instantaneous phase difference (IPD) and envelope difference between channels
based on the analytic signal representation.

.. code-block:: python

   import audiotoolbox as audio

   sig = audio.Signal(n_channels=2, duration=1, fs=48000)
   sig.ch[0].add_tone(1000)
   sig.ch[1].add_tone(1000, start_phase=0.3)

   ipd, ild = audio.extract_binaural_differences(sig)

Notes
-----

- These methods are most meaningful for narrowband or band-limited signals.
- For broadband stimuli, analyze in subbands (for example with a filterbank)
  before interpreting phase/coherence metrics.
