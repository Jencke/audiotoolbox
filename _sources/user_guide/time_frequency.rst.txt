
Time-Frequency Methods
======================

Spectrograms
------------

The time-frequency submodule provides several methods of calculating spectrograms either based on short-term Fourier transforms or based on filterbanks. 


Plotting a gamma-tone filterbank (1/3 ERB spacing) based spectrogram of a 500Hz tone in pink noise:

.. plot::
    :include-source:

    sig = audio.Signal(1, 1, 48000)
    sig.add_tone(500).set_dbfs(0)
    sig.add_noise("pink")
    sig.add_fade_window(10e-3)


    spec, fc = sig.time_frequency.gammatone_specgram(
        nperseg=1024, noverlap=512, flow=16, fhigh=16000, step=1 / 3
    )    
    fig, ax = plt.subplots(1, 1)
    cb = ax.pcolormesh(spec.time, fc, spec.T)
    ax.set_yscale("log")
    ax.set_ylim(16, 16000)
    ax.set_ylabel("Frequency / Hz")
    ax.set_xlabel("Time / s")
    ax.set_title("Gammatone spectrogram")
    cb = plt.colorbar(cb, ax=ax)
    cb.set_label("dB FS")
    plt.show()

The same signal in an octave-band based spectrogram:

.. plot::
    :include-source:

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


It's also possible to create spectrograms based on custom filter-banks

.. plot::
    :include-source:

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
