Develop branch
 - Deleted the long depricated Signal.add_noise_noise method
 - Implemented Signal.stats submodule
 - Added deprecationwarning to Signal.calc_dbfs and Signal.calc_dbspl which
   moved to the Signal.stats submodule
 - Added the option to directly  apply filters when generating partly correlated noise
 - Added the audiotools.cmplx_corr function which calculates the
   complex-valued correlation coefficent
 - Renamed calc_coherence to cmplx_crosscorr

0.61 -> 0.62
 - Fixed the shape of the channels after appling a filterbank
 - Added a summary method that prints information about size and shape of the object
