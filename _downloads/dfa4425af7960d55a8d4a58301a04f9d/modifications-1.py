import audiotoolbox as audio
import numpy as np
import matplotlib.pyplot as plt

sig = audio.Signal(n_channels=2, duration=20e-3, fs=48000)
sig.ch[0].add_tone(frequency=500, amplitude=1, start_phase=0)
sig.ch[1].add_tone(frequency=500, amplitude=1, start_phase=np.pi)

plt.plot(sig.time * 1e3, sig)
plt.xlabel('Time / ms')
plt.ylabel('Amplitude')
plt.title('Antiphasic 500Hz Tones')
plt.grid(True)
plt.show()