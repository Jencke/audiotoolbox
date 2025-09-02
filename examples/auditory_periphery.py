from . import audiotoolbox as audio
import numpy as np
import matplotlib.pyplot as plt

# create a 200ms long noise token with 10ms cosine ramps
sig = audio.Signal(1, 200e-3, 48000).add_noise()
sig.add_fade_window(10e-3, "cos")

# Calculate filter center frequencies form 100Hz to 1kHz equally
# spaced on ther ERB scale (1 Filter / ERB)
fc_vec = audio.freqarange(100, 4000, 3, "erb")
bw_vec = audio.calc_bandwidth(fc_vec, "erb")

# Create a complex gammatone filter bank and apply it to the signal
bank = audio.create_filterbank(fc_vec, bw_vec, "gammatone", sig.fs)
sig_out = bank.filt(sig)

# envelope compression by taking the envelope to a power of 0.24
sig_out = np.abs(sig_out) ** 0.24 * np.exp(1j * np.angle(sig_out))
sig_out = np.real(sig_out).rectify()


# Plot the results
fig, ax = plt.subplots(3, 3, figsize=(7, 5))
ax = ax.flatten()
for i_f, f in enumerate(fc_vec[:9]):
    res, _ = audio.filter.butterworth(sig_out.ch[i_f], None, 400, sig_out.fs)
    ax[i_f].plot(sig_out.time * 1e3, res)
    ax[i_f].set_title(f"$f_c$={f:.1f} Hz")
    ax[i_f].set_xlabel("Time / ms")
    ax[i_f].set_ylabel("Amplitude")
fig.subplots_adjust(
    top=0.95, bottom=0.09, left=0.08, right=0.98, hspace=0.9, wspace=0.4
)
