import audiotoolbox as audio
import matplotlib.pyplot as plt

sig = audio.Signal(n_channels=1, duration=500e-3, fs=48000)
sig.add_tone(1000)
sig.add_cos_modulator(frequency=30, m=1)
sig.add_fade_window(100e-3)

plt.plot(sig.time * 1e3, sig)
plt.xlabel('Time / ms')
plt.ylabel('Amplitude')
plt.title('1kHz Tone with 30Hz Modulator')
plt.grid(True)
plt.show()