sig = audio.Signal(1, 1, 48000)
sig.add_tone(500).set_dbfs(0)
sig.add_noise("pink")
sig.add_fade_window(10e-3)

bank = audio.filter.bank.create_filterbank(fc=[250, 500, 1000], bw=[25, 50, 100], filter_type="butter", fs=sig.fs)


spec, fc = sig.time_frequency.filterbank_specgram(bank=bank,
    nperseg=1024, noverlap=512
)
fig, ax = plt.subplots(1, 1)
cb = ax.pcolormesh(spec.time, fc, spec.T)
ax.set_yscale("log")
ax.set_ylim(100, 1500)
ax.set_ylabel("Frequency / Hz")
ax.set_xlabel("Time / s")
ax.set_title("Custom filter-bank spectrogram")
cb = plt.colorbar(cb, ax=ax)
cb.set_label("dB FS")
plt.show()