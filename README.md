# audiotools

A toolbox for generating acoustic stimuli primariliy aimed at
auditory research and psychoacoustics.

## Audiotools provides two interfaces.

### Object oriented interface (Oaudio)

Oaudio provides an easy to use fluent interface to most of the
functionality implemented in the function based interface.

```python
stim = Signal().init_signal(1, 1, 48000)
sitm.add_noise()
stim.bandpass(500, 200, 'brickwall')
stim.set_dbspl(60).add_fade_window(50e-3, 'cos')
```

Creates a one channel, 1 second long noise stimulus centered at 500Hz
with a bandwidth of 200Hz at 48kHz sampling rate. The amplitude is set
to be equivelent to 60dB (SPL) and the noise is ramped in and out
using 50ms raised cosine slopes.

### Function based interface.

This interface provides functions to generate stimuli direktly on the
basis of numpy arrays. It also provides a set of functions to modify
or analyze these stimuli.
