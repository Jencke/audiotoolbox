## 1.0 -> 1.10

### Added

- Spectrum plotting in the `Visualization` sub-class.
- Complex exponential filter support.
- Explicit guidance for complex dtype workflows: `complex_signal = signal.astype(complex)`.
- `BaseSignal.channel_shape` property: always returns a `tuple` of the channel axes shape (e.g. `(1,)` for mono, `(2,)` for stereo). Use this in generic shape-building code instead of `n_channels`, which stays user-friendly as an `int`.

### Changed

- `BaseSignal` now always keeps an explicit channel axis. Mono signals are represented as `(n_samples, 1)`.
- `Signal.bandpass(..., return_complex=True)` now returns a new complex `Signal` instead of attempting in-place dtype mutation.

### Fixed

- `SignalStats.octave_band_levels` return order is now `(frequencies, levels)`.
- `Visualization.specgram_overview` now unpacks octave-band outputs correctly.
- `Visualization.spectrum` now correctly applies `minx` and `maxx` via `ax.set_xlim(...)`.
- `Visualization.spectrum` now uses `10*log10` for `power=True, in_db=True` and `20*log10` for amplitude.
- NumPy deprecation warnings in filterbank tests by explicitly extracting scalar values.

### Deprecated

- `Signal.bandpass(..., return_complex=True)` now emits a `UserWarning` to make complex-output behavior explicit.


## 0.75 -> 1.0

### Added

- `specgram_overview`, which plots a 1/3-octave spectrogram, the time signal, and overall 1/3-octave band levels.
- Instantaneous complex correlation with a sliding window.

### Changed

- Major refactoring of the codebase for better maintainability.
- Plotting functionality moved into the `Signal.viz` submodule.
- `Signal.plot` moved to `Signal.viz.plot`.

### Fixed

- Bug that could result in incorrect dtypes when convolving.

### Deprecated

- Old, unused function interface.

## 0.74 -> 0.75

### Added

- `play` method for quick signal playback.
- `resample` method on `Signal`.

### Changed

- Major restructuring of the `Signal` class (fully backward compatible).
- `clip` renamed to `trim` to avoid conflicting with the NumPy method.

### Documentation

- Improved documentation.

## 0.73 -> 0.74

### Added

- Several spectrograms in the `time_frequency` submodule of `Signal`.
- `Signal.as_blocked`, generating a blocked view on the original signal.

### Changed

- Octave-band defaults now use preferred frequencies.

### Documentation

- Improved documentation.

## 0.72 -> 0.73

### Added

- `octave_band_levels` method in the `Signal.stats` submodule.

### Fixed

- Bug that caused linear crossfade inversion.

### Removed

- Deprecated `calc_dbfs` from `Signal`.

## 0.70 -> 0.72

### Changed

- `writefile` renamed to `write_file`.
- `wav` submodule renamed to `file_io`.
- `rms` moved into the `stats` submodule.

### Documentation

- Improved documentation.

## 0.68 -> 0.70

### Added

- `crossfade` function.
- `convolve` method on `Signal`.
- Options to define the first and last sample to read from file.
- `info` function to extract audio file metadata without reading full sample data.

### Changed

- Library renamed to `audiotoolbox`.
- Ongoing refactoring of the auditory-scales structure.

## 0.67 -> 0.68

### Changed

- Reading and writing audio files moved to the `soundfile` library.

### Fixed

- Test fixes and compatibility fixes for deprecated NumPy APIs.

## 0.66.1 -> 0.67

### Added

- A-weighting and C-weighting functions.
- `add_gain` method on the `Signal` class.

### Changed

- `stats` functions converted to properties.

## 0.66 -> 0.66.1

### Changed

- Type hint updated for better backward compatibility.

## 0.65.1 -> 0.66

### Added

- Option to generate partly correlated noise with different spectral shapes.

## 0.65 -> 0.65.1

### Changed

- Filterbanks became indexable for easier access to individual filters.

### Fixed

- Small bug in the ERB-to-3dB conversion for gammatone filters that could result in a type error.

## 0.64.1 -> 0.65

### Added

- Default auditory gammatone filterbank.
- Default fractional-octave filterbank.
- Function to convert frequency into band number following the ANSI norm.
- Octave spacing support in `audiotools.freqarange`.

### Changed

- Filterbanks moved from `audiotools.filter` to `audiotools.filter.bank`.

### Fixed

- Small bug in `_copy_to_dim` that could remove the last dimension of an array if it equaled 1.

### Removed

- Broken audio playback functionality.

## 0.64 -> 0.64.1

### Fixed

- Bug in `FrequencyDomainSignal.to_timedomain()`.

## 0.62 -> 0.64

### Added

- Parameter in `audiotools.dbfs` to specify whether dB full scale is relative to peak or RMS.
- `Signal.stats` submodule.
- Option to directly apply filters when generating partly correlated noise.
- `audiotools.cmplx_corr` for complex-valued correlation coefficients.
- `DeprecationWarning` for `Signal.calc_dbfs` and `Signal.calc_dbspl`, which moved to `Signal.stats`.

### Changed

- `calc_coherence` renamed to `cmplx_crosscorr`.

### Fixed

- Bug in `signal.phase_shift` that could produce a complex-valued signal unexpectedly.

### Removed

- Long-deprecated `Signal.add_noise_noise` method.

## 0.61 -> 0.62

### Added

- `summary` method that prints object size and shape information.

### Fixed

- Channel shape handling after applying a filterbank.

## 0.57 -> 0.6

### Changed

- Return values in `octave_band_levels` swapped for consistency.

### Documentation

- Improved documentation.
