In Developement:
 1. New Features
   * Implemented new add_uncorr_noise and generate_uncorr_noise methods
	 to generate/add partly uncorrelated noise to the signal
   * nsamples, generate_noise,  now all accepts signal class instead of duration and fs
   * generate_noise and generate_low_noise_noise can now create noise
     tokes with arbitrary dimensions

 2.Compatibility breaking:
   * Changed order of Parameters in generate_low_noise_noise
   * Removed add_corr_noise and generate_corr_noise methods -> use
     add_uncorr_noise instead
   * Gammatone filter now defaults to interpreting the bw parameter as
     ERB not as -3dB bandwidth - can be chaged by setting
     attenuation_db=-3
