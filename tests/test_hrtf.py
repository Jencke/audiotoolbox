import audiotoolbox as audio
import numpy as np
import numpy.testing as testing
import pytest


def _make_hrir_set(n_taps=64, fs=48000):
    """A simple horizontal-plane HRIR set with four cardinal directions."""
    azimuths = np.array([0, 90, 180, 270])
    positions = np.stack([azimuths, np.zeros_like(azimuths)], axis=1)
    hrirs = audio.Signal((len(azimuths), 2), n_taps / fs, fs)
    # Give every direction/ear a distinct impulse so we can track it.
    for i in range(len(azimuths)):
        hrirs.ch[i, 0][i] = 1.0  # left ear: impulse at sample i
        hrirs.ch[i, 1][i + 1] = 1.0  # right ear: impulse at sample i+1
    return audio.HRIRSet(hrirs, positions), positions


def test_construction_and_properties():
    hrir_set, positions = _make_hrir_set()
    assert hrir_set.n_directions == 4
    assert hrir_set.n_taps == 64
    assert hrir_set.fs == 48000
    testing.assert_array_equal(hrir_set.azimuth, positions[:, 0])
    testing.assert_array_equal(hrir_set.elevation, positions[:, 1])
    assert hrir_set.distance is None


def test_construction_validates_shape():
    fs = 48000
    # last channel axis must be 2 (left, right)
    bad = audio.Signal((4, 3), 64 / fs, fs)
    with pytest.raises(ValueError):
        audio.HRIRSet(bad, np.zeros((4, 2)))

    # direction count must match position count
    ok = audio.Signal((4, 2), 64 / fs, fs)
    with pytest.raises(ValueError):
        audio.HRIRSet(ok, np.zeros((3, 2)))


def test_nearest_returns_exact_direction():
    hrir_set, _ = _make_hrir_set()
    # query close to the 90 deg measurement -> direction index 1
    hrir = hrir_set.nearest(85, 0)
    assert hrir.shape == (64, 2)
    assert hrir.ch[0][1] == 1.0  # left-ear impulse for index 1 sits at sample 1


def test_interpolate_between_measurements():
    hrir_set, _ = _make_hrir_set()
    # halfway between 0 deg (idx 0) and 90 deg (idx 1) on the horizontal plane
    hrir = hrir_set.interpolate(45, 0)
    assert hrir.shape == (64, 2)
    left = np.asarray(hrir)[:, 0].ravel()
    # both contributing impulses present, weights sum to 1
    assert left[0] > 0 and left[1] > 0
    testing.assert_allclose(left[0] + left[1], 1.0, atol=1e-6)


def test_interpolate_full_sphere_barycentric():
    # a non-coplanar grid forces the 3-D ConvexHull barycentric path
    fs = 48000
    n_taps = 16
    positions = np.array(
        [
            [0, 0], [90, 0], [180, 0], [270, 0],  # horizontal ring
            [0, 90], [0, -90],                     # poles -> full 3-D cloud
        ],
        dtype=float,
    )
    hrirs = audio.Signal((len(positions), 2), n_taps / fs, fs)
    for i in range(len(positions)):
        hrirs.ch[i, 0][i] = 1.0
    hrir_set = audio.HRIRSet(hrirs, positions)

    # a direction inside the triangle (0,0)-(90,0)-(0,90)
    hrir = hrir_set.interpolate(30, 30)
    assert hrir.shape == (n_taps, 2)
    left = np.asarray(hrir)[:, 0].ravel()
    # weights are a partition of unity over the contributing impulses
    testing.assert_allclose(left.sum(), 1.0, atol=1e-6)
    # an exact measured direction returns that HRIR unchanged
    exact = hrir_set.interpolate(90, 0)
    assert np.asarray(exact)[1, 0] == 1.0


def test_to_hrtf_roundtrip():
    hrir_set, _ = _make_hrir_set()
    hrtf = hrir_set.to_hrtf()
    assert isinstance(hrtf, audio.FrequencyDomainSignal)
    back = hrtf.to_timedomain()
    testing.assert_allclose(np.asarray(back), np.asarray(hrir_set.hrirs), atol=1e-9)


def test_render_produces_binaural_and_preserves_input():
    hrir_set, _ = _make_hrir_set(n_taps=8)
    src = audio.Signal(1, 100 / 48000, 48000).add_noise(seed=1)
    src_before = src.copy()

    out = hrir_set.render(src, 90, 0, interpolate=False)

    assert out.n_channels == 2
    assert out.n_samples == src.n_samples + hrir_set.n_taps - 1
    # caller's signal must be untouched (convolve mutates in place internally)
    testing.assert_array_equal(np.asarray(src), np.asarray(src_before))

    # rendering with the nearest (idx 1) HRIR is a pure delay of 1 / 2 samples
    left = np.asarray(out)[:, 0]
    testing.assert_allclose(left[1 : 1 + src.n_samples].ravel(),
                            np.asarray(src).ravel(), atol=1e-9)


def test_render_rejects_non_mono():
    hrir_set, _ = _make_hrir_set()
    stereo = audio.Signal(2, 100 / 48000, 48000)
    with pytest.raises(ValueError):
        hrir_set.render(stereo, 0, 0)


def test_from_sofa_roundtrip(tmp_path):
    sofar = pytest.importorskip("sofar")

    fs = 48000
    n_taps = 16
    azimuths = np.array([0.0, 90.0, 180.0, 270.0])
    n_dir = len(azimuths)

    sofa = sofar.Sofa("SimpleFreeFieldHRIR")
    ir = np.zeros((n_dir, 2, n_taps))
    for i in range(n_dir):
        ir[i, 0, i] = 1.0
        ir[i, 1, i + 1] = 1.0
    sofa.Data_IR = ir
    sofa.Data_SamplingRate = fs
    sofa.SourcePosition = np.stack(
        [azimuths, np.zeros(n_dir), np.ones(n_dir)], axis=1
    )
    sofa.GLOBAL_RoomType = "free field"

    path = tmp_path / "test.sofa"
    sofar.write_sofa(str(path), sofa)

    hrir_set = audio.HRIRSet.from_sofa(str(path))
    assert hrir_set.n_directions == n_dir
    assert hrir_set.n_taps == n_taps
    assert hrir_set.fs == fs
    testing.assert_array_equal(hrir_set.azimuth, azimuths)
    # IR reordered to (n_taps, n_directions, 2); check a known impulse
    assert hrir_set.hrirs.ch[1, 0][1] == 1.0
