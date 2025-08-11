sig = audio.Signal(2, 20e-3, 48000)
sig.ch[0].add_tone(frequency=500, amplitude=1, start_phase=0)
sig.ch[1].add_tone(frequency=500, amplitude=1, start_phase=np.pi)

plt.plot(sig.time * 1e3, sig)
plt.xlabel('Time / ms')
plt.ylabel('Amplitude')
plt.title('Antiphasic 500Hz tones')
plt.show()