white_noise = audio.Signal(1, 1, 48000).add_noise()
pink_noise = audio.Signal(1, 1, 48000).add_noise(ntype='pink')
brown_noise = audio.Signal(1, 1, 48000).add_noise(ntype='brown')

wspec, fc = white_noise.time_frequency.octave_band_specgram(oct_fraction=3)
pspec, fc = pink_noise.time_frequency.octave_band_specgram(oct_fraction=3)
bspec, fc = brown_noise.time_frequency.octave_band_specgram(oct_fraction=3)

norm = plt.Normalize(min([wspec.min(), pspec.min(), bspec.min()]), max([wspec.max(), pspec.max(), bspec.max()]))
fig, ax = plt.subplots(2, 2, sharex='all', sharey='all')
ax[0, 0].set_title('White Noise')
ax[0, 0].pcolormesh(wspec.time, fc, wspec.T, norm=norm)
ax[0, 1].set_title('Pink Noise')
ax[0, 1].pcolormesh(pspec.time, fc, pspec.T, norm=norm)
ax[1, 0].set_title('Brown Noise')
ax[1, 0].pcolormesh(bspec.time, fc, bspec.T, norm=norm)


ax[1, 0].set_xlabel("Time / s")
for a in ax[:, 0]:
    a.set_ylabel('Frequency / Hz')

for a in ax.flatten():
    a.set_yscale('log')
ax[1, 1].set_visible(False)