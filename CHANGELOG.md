## 1.11 -> 1.12

### Added

- `Signal.add_uncorr_noise` now supports a negative `corr` for two channels, realised by sign-inverting one channel. For more than two channels (where a uniform negative correlation is not achievable) the positive magnitude is used and a `UserWarning` is emitted.
- `HRIRSet` class (`audiotoolbox.HRIRSet`) for holding head-related impulse responses measured over directions. The impulse responses are stored as a `Signal` of shape `(n_taps, n_directions, 2)` alongside a source-position table, following the library's composition pattern.
- `HRIRSet.from_sofa(...)` to load HRIRs from SOFA files (the standard HRTF interchange format) via the optional `sofar` dependency.
- Direction lookup via `HRIRSet.nearest(...)` and direction interpolation via `HRIRSet.interpolate(...)`: barycentric over the surrounding spherical triangle for fully three-dimensional measurement grids, and angular interpolation between adjacent directions for coplanar grids (e.g. a horizontal ring).
- `HRIRSet.render(...)` to spatialize a mono signal into a binaural signal by convolving it with the (left, right) HRIR for a requested direction.
- `HRIRSet.to_hrtf()` returning the frequency-domain transfer functions as a `FrequencyDomainSignal`.
- New optional dependency extra `hrtf` (installs `sofar`); use `pip install audiotoolbox[hrtf]` for SOFA file support.
- `Signal.remove_silence` now supports `edges_only=True` to remove only leading and trailing silence while preserving silent gaps inside the kept region.

### Changed

- `Signal.to_analytical()` now uses `scipy.signal.hilbert(..., axis=0)` for real-valued signals instead of round-tripping through the frequency-domain representation, which substantially reduces runtime for common real-signal cases.
- `Signal.ch[...]` now normalizes single-channel selections back to the library's canonical mono shape `(n_samples, 1)` instead of collapsing them to a 1-D array when indexing multidimensional channel layouts.
- `Signal.ch[...]` now handles slices and ellipsis consistently across channel axes and raises `IndexError` for invalid channel indices instead of relying on NumPy's less explicit indexing quirks.

### Fixed
- `Signal.add_noise` now consistently adds noise to the existing signal for all spectral shapes; previously the `white` branch overwrote the signal content instead of adding to it.
- `Signal.add_uncorr_noise` now produces independent noise tokens when a `seed` is given. Previously each channel was reseeded with the same value, so all tokens were identical and the orthogonalization left all but one channel as a degenerate (non-noise) signal. `Signal.add_noise` now only reseeds the RNG when a seed is explicitly provided.
- `Signal.convolve` now accepts a plain `ndarray` kernel as documented, instead of raising `AttributeError`.
- `Signal.convolve` with a complex kernel no longer silently discards the imaginary part; the output dtype is promoted and a new complex `Signal` is returned when the input is real (mirroring `Signal.bandpass`).
- `Signal.convolve` no longer raises a broadcasting `ValueError` when a trailing singleton channel axis takes part in the overlapping dimensions; the overlap is now determined after squeezing such axes.
- `Signal.remove_silence` no longer emits expected internal warnings during silence analysis (block zero-padding and dBFS divide-by-zero for silent blocks).
- `Signal.to_analytical()` now preserves support for complex-valued inputs by falling back to the previous frequency-domain implementation when the input signal is already complex.
- Correlation helpers that duplicate mono channels internally now accept canonical mono `Signal.ch[...]` views directly instead of assuming channel selections collapse to 1-D arrays.

## 1.10 -> 1.11

### Added

- New class-based auditory scales API in `audiotoolbox.scales` with `BarkScale`, `ErbScale`, and `OctaveScale`.
- Unified scale method surface across scales: `from_freq(...)`, `to_freq(...)`, and `get_bw(...)` (`calc_bw(...)` kept as a compatibility alias).
- Ready-to-use scale instances exported as `audio.bark`, `audio.erb`, and `audio.octave`.
- Added `MelScale`, `SemitoneScale`, and `GreenwoodScale` with top-level instances `audio.mel`, `audio.semitone`, and `audio.greenwood`.

### Changed

- Scale conversion and bandwidth logic has been moved into the dedicated `scales` submodule classes; core scale helpers now delegate to those implementations.
- Scale APIs now consistently accept Python lists and NumPy arrays for conversions and bandwidth calculations.

### Fixed

- Bark scale input range checks now raise explicit `ValueError`s instead of relying on `assert`.
- Octave scale validation now rejects invalid `oct_fraction` values with clear errors.
- `BarkScale.get_bark_limits()` now returns a copy to avoid accidental mutation of internal lookup data.

### Deprecated

- Core scale wrapper functions are now deprecated in favor of the scale object API:
	`get_bark_limits`, `bark_to_freq`, `freq_to_bark`, `freq_to_erb`,
	`erb_to_freq`, `freq_to_octband`, `octband_to_freq`, and `calc_bandwidth`.

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
