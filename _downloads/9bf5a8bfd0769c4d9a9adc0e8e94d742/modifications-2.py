import audiotoolbox as audio
import matplotlib.pyplot as plt

sig = audio.Signal(n_channels=1, duration=100e-3, fs=48000)
sig.add_tone(frequency=500, amplitude=1, start_phase=0)
sig.add_fade_window(rise_time=30e-3, win_type='cos')

plt.plot(sig.time * 1e3, sig)
plt.xlabel('Time / ms')
plt.ylabel('Amplitude')
plt.title('Tone with Raised Cosine Fade-in and -out')
plt.grid(True)
plt.show()