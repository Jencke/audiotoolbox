import audiotoolbox as audio
import numpy as np
import matplotlib.pyplot as plt

# Create a pink noise signal
noise = audio.Signal(1, duration=5, fs=48000).add_noise('white')

# Calculate octave-band levels
fc, levels = noise.stats.octave_band_levels(oct_fraction=3)

base_value = -50
# Plot the results
plt.figure(figsize=(8, 5))
plt.bar(np.arange(len(fc)), levels -base_value, tick_label=np.round(fc).astype(int), bottom=base_value)
# plt.bar(np.arange(len(fc)), levels, tick_label=np.round(fc).astype(int))
plt.title('1/3-Octave Band Levels of White Noise')
plt.xlabel('Center Frequency / Hz')
plt.ylabel('Level / dBFS')
plt.xticks(rotation=-45)
plt.tight_layout()
plt.show()