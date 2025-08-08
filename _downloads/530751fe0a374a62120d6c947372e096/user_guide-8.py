sig = audio.Signal(1, 1, 48000)
sig.add_tone(500).set_dbfs(0)
sig.add_noise("pink")
sig.add_fade_window(10e-3)


spec, fc = sig.time_frequency.octave_band_specgram(
    nperseg=1024, noverlap=512, flow=16, fhigh=16000
)
fig, ax = plt.subplots(1, 1)
cb = ax.pcolormesh(spec.time, fc, spec.T)
ax.set_yscale("log")
ax.set_ylim(16, 16000)
ax.set_ylabel("Frequency / Hz")
ax.set_xlabel("Time / s")
ax.set_title("1/3 Octave-band spectrogram")
cb = plt.colorbar(cb, ax=ax)
cb.set_label("dB FS")
plt.show()