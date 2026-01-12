sig = (
audio.Signal(1, 1, 48000)
.add_noise("pink")
.bandpass(500, 500, "butter", order=2)
.add_cos_modulator(6, 1)
.set_dbfs(-10)
)
fig, ax = sig.viz.specgram_overview()
fig.show()