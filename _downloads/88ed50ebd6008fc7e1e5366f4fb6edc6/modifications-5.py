import audiotoolbox as audio
import matplotlib.pyplot as plt
import numpy as np

# Create two distinct signals
sig1 = audio.Signal(1, 0.5, 48000).add_tone(400).set_dbfs(-30)
sig2 = audio.Signal(1, 0.5, 48000).add_tone(400).set_dbfs(-20)

# Crossfade them with a 100ms linear fade
fade_duration = 100e-3
crossfaded_sig = audio.crossfade(sig1, sig2, fade_duration, fade_type='linear')

# --- Plotting ---
fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True, sharey=True)

# Plot original signals for context
ax[0].plot(sig1.time, sig1, color='C0')
ax[0].set_title('Signal 1 (400 Hz)')
ax[0].grid(True)

# Shift time axis for the second signal to show its original position
time_sig2 = sig2.time + sig1.duration - fade_duration
ax[1].plot(time_sig2, sig2, color='C1')
ax[1].set_title('Signal 2 (800 Hz)')
ax[1].grid(True)

# Plot the final crossfaded signal
ax[2].plot(crossfaded_sig.time, crossfaded_sig, color='C2')
ax[2].set_title('Crossfaded Signal')
ax[2].set_xlabel('Time / s')
ax[2].grid(True)

# Highlight the crossfade region
fade_start_time = sig1.duration - fade_duration
ax[2].axvspan(fade_start_time, sig1.duration, color='black', alpha=0.15, label='Crossfade Region')
ax[2].legend(loc='upper right')

for a in ax:
   a.label_outer()

plt.tight_layout()
plt.show()