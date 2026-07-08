.. _guide_hrtf:

Head-Related Transfer Functions
================================

A head-related transfer function (HRTF), or its time-domain counterpart
the head-related impulse response (HRIR), describes how a sound arriving
from a given direction is filtered by the head, torso, and outer ears
before it reaches the eardrums. Convolving a mono source with the
left/right HRIR for a direction places that source at the corresponding
position in a binaural (two-channel) signal.

The :class:`audiotoolbox.HRIRSet` class holds a set of HRIRs measured over
many directions and provides direction lookup, interpolation, and
rendering. The impulse responses are stored as a regular
:class:`audiotoolbox.Signal` of shape ``(n_taps, n_directions, 2)`` (the
last axis being the left and right ear), while the measured directions are
kept in a separate position table.

Loading a SOFA file
--------------------

SOFA (Spatially Oriented Format for Acoustics) is the standard interchange
format for HRTFs. :meth:`audiotoolbox.HRIRSet.from_sofa` reads such files
using the optional `sofar <https://pypi.org/project/sofar/>`_ package,
which you can install with::

    pip install audiotoolbox[hrtf]

>>> import audiotoolbox as audio
>>> hrir_set = audio.HRIRSet.from_sofa('subject_003.sofa')
>>> hrir_set.n_directions
1250
>>> hrir_set.fs
44100

The measured directions are available as the ``azimuth`` and ``elevation``
attributes (in degrees, following the SOFA convention where azimuth
increases counter-clockwise and elevation is measured from the horizontal
plane).

Building a set manually
-----------------------

An ``HRIRSet`` can also be constructed directly from a ``Signal`` and a
table of source positions given as ``(azimuth, elevation[, distance])``.

>>> import numpy as np
>>> positions = np.array([[0, 0], [90, 0], [180, 0], [270, 0]])
>>> hrirs = audio.Signal((4, 2), 128 / 48000, 48000)  # 4 directions, L/R
>>> hrir_set = audio.HRIRSet(hrirs, positions)
>>> hrir_set.n_directions
4

Looking up and interpolating directions
----------------------------------------

:meth:`audiotoolbox.HRIRSet.nearest` returns the HRIR of the closest
measured direction as a two-channel ``Signal``, while
:meth:`audiotoolbox.HRIRSet.interpolate` synthesizes an HRIR for an
arbitrary direction. For a fully three-dimensional measurement grid the
three directions surrounding the query are combined using barycentric
weights on the sphere; for a planar grid (such as a horizontal ring) the
two adjacent directions are interpolated instead. Directions outside the
measured region fall back to the nearest measurement.

>>> hrir = hrir_set.interpolate(45, 0)
>>> hrir.shape
(128, 2)

Rendering a source
------------------

:meth:`audiotoolbox.HRIRSet.render` spatializes a mono signal by convolving
it with the (left, right) HRIR for the requested direction. The input
signal is left untouched and a new two-channel signal is returned.

>>> source = audio.Signal(1, 1, 48000).add_noise()
>>> binaural = hrir_set.render(source, azimuth=45, elevation=0)
>>> binaural.n_channels
2

Set ``interpolate=False`` to use the nearest measured HRIR instead of an
interpolated one.

Working in the frequency domain
-------------------------------

:meth:`audiotoolbox.HRIRSet.to_hrtf` returns the frequency-domain transfer
functions as a :class:`audiotoolbox.FrequencyDomainSignal`. As with
:meth:`audiotoolbox.Signal.to_freqdomain`, this is not done in place; a new
object is returned.

>>> hrtf = hrir_set.to_hrtf()
>>> hrtf.n_samples
128
