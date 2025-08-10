import audiotoolbox as audio
import matplotlib.pyplot as plt
import numpy as np

# Create a white noise signal
sig = audio.Signal(n_channels=1, duration=100e-3, fs=48000).add_noise('white')

# Create a low-passed version of the signal
# A copy is made so the original signal is not modified
lp_sig = sig.copy().lowpass(f_cut=1000, filter_type='butter', order=4)

# Plot the original and filtered signals
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 6))
ax[0].plot(sig.time, sig, label='Original')
ax[0].set_title('Original White Noise')
ax[0].grid(True)

ax[1].plot(lp_sig.time, lp_sig, label='Filtered', color='C1')
ax[1].set_title('After 1kHz Low-Pass Filter')
ax[1].set_xlabel('Time / s')
ax[1].grid(True)

for a in ax:
    a.set_ylabel('Amplitude')

plt.tight_layout()
plt.show()