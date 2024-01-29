0.67 -> 0.68
 - Reading and Writing Audiofiles is now handled using the soundfile library

0.66.1 -> 0.67
 - Changed functions in stats to property
 - Implemented functions for A and C frequency weighting
 - Implemented an add_gain method to the signal class

0.66 -> 0.66.1
 - Changed a type hint for better backward compatibility

0.65.1 -> 0.66
 - Added option to generate partly correlated noise with different spectral shapes

0.65 -> 0.65.1
 - Made filterbanks indexable to gain excess to individual filters
 - fixed a small bug in the erb to 3db calculation function of the gammatone
   filter which could result in a type error

0.64.1 -> 0.65
 - Added a default auditory gammatone filterbank
 - Added a default fractional octave filterbank
 - Added function to convert frequency into band number following ANSI norm
 - Fixed a small bug in _copy_to_dim which would remove the last dimension of an error if it equaled 1
 - Deleted broken audio playback functionality
 - Moved filterbanks from audiotools.filter to audiotools.filter.bank submodule
 - audiotools.freqarange now supports octave spacing

0.64 -> 0.64.1
 - Fixed a bug in the FrequencyDomainSignal.to_timedomain() method

0.62 -> 0.64
 - Added parameter to auditools.dbfs to specify if dB fullscale is relative to peak or rms level
 - Deleted the long depricated Signal.add_noise_noise method
 - Implemented Signal.stats submodule
 - Added deprecationwarning to Signal.calc_dbfs and Signal.calc_dbspl which
   moved to the Signal.stats submodule
 - Added the option to directly  apply filters when generating partly correlated noise
 - Added the audiotools.cmplx_corr function which calculates the
   complex-valued correlation coefficent
 - Renamed calc_coherence to cmplx_crosscorr
 - Fixed a bug in signal.phase_shift that sometimes resulted in a complex valued signal

0.61 -> 0.62
 - Fixed the shape of the channels after appling a filterbank
 - Added a summary method that prints information about size and shape of the object
