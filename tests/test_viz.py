import numpy as np
import matplotlib

matplotlib.use("Agg")

import audiotoolbox as audio


def test_spectrum_applies_x_limits():
    sig = audio.Signal(1, 1, 48000).add_tone(1000)
    _, ax = sig.viz.spectrum(minx=100, maxx=2000)

    xlim = ax.get_xlim()
    assert xlim[0] == 100
    assert xlim[1] == 2000


def test_spectrum_power_in_db_uses_10log10():
    sig = audio.Signal(1, 1, 48000).add_tone(1000)

    _, ax_amp = sig.viz.spectrum(single_sided=False, power=False, in_db=True)
    _, ax_pow = sig.viz.spectrum(single_sided=False, power=True, in_db=True)

    amp_y = np.asarray(ax_amp.lines[0].get_ydata())
    pow_y = np.asarray(ax_pow.lines[0].get_ydata())

    # Validate the peak bin where numerical flooring has negligible impact.
    peak_idx = int(np.argmax(amp_y))
    np.testing.assert_allclose(pow_y[peak_idx], amp_y[peak_idx], atol=1e-8, rtol=0)
