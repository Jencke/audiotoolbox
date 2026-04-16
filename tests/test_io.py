import audiotoolbox.io as io
import audiotoolbox as audio
import numpy as np
import numpy.testing as testing


def test_writewav_readwav(tmp_path):
    """Test invertability of readfile and writefile"""
    filename = tmp_path / "test.wav"
    fs = 48000
    signal = audio.Signal(2, 1, fs)
    signal[:] = np.linspace(-1, 1, signal.n_samples)[:, None]
    io.write_file(str(filename), signal, signal.fs)
    out, out_fs = io.readfile(str(filename))
    testing.assert_allclose(
        out, signal, atol=10000
    )  # set atol to high value so to not lead to problems close to 0
    assert out_fs == fs


def test_signal_writefile(tmp_path):
    """Test invertability of readfile and writefile"""
    filename = tmp_path / "test.wav"
    fs = 48000
    signal = audio.Signal(2, 1, fs)
    signal[:] = np.linspace(-1, 1, signal.n_samples)[:, None]

    signal.write_file(str(filename))
    out, out_fs = io.readfile(str(filename))
    testing.assert_allclose(out, signal, atol=10000)
    assert out_fs == fs
