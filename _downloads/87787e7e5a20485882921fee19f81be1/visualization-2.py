sig = (
audio.Signal(1, 1, 48000)
.add_noise("pink")
.bandpass(1000, 500, "butter", order=2)
.add_cos_modulator(6, 1)
.set_dbfs(-10)
)

fig, ax = sig.viz.spectrum(single_sided=True, in_db=True)
fig.show()