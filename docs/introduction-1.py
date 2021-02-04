import audiotools as audio
import matplotlib.pyplot as plt
sig = audio.Signal(n_channels=1, duration=100e-3, fs=48000)
sig.add_tone(500).set_dbspl(60).add_fade_window(10e-3, 'cos')
plt.title('100ms long 500Hz tone with raised cosine slopes')
plt.plot(sig.time, sig)
plt.xlabel('Time / s')
plt.ylabel('Amplitude')
plt.show()
