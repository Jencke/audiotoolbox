import numpy as np
import numpy.testing as testing
import pytest

import audiotoolbox as audio


def test_scale_modules_expose_consistent_api():
	for scale in (audio.bark, audio.erb, audio.octave):
		assert hasattr(scale, "from_freq")
		assert hasattr(scale, "to_freq")
		assert hasattr(scale, "get_bw")
		assert hasattr(scale, "calc_bw")


def test_scale_instances_expose_consistent_api():
	for scale in (
		audio.scales.bark_scale,
		audio.scales.erb_scale,
		audio.scales.octave_scale,
	):
		assert hasattr(scale, "from_freq")
		assert hasattr(scale, "to_freq")
		assert hasattr(scale, "get_bw")
		assert hasattr(scale, "calc_bw")


def test_bark_scalar_roundtrip():
	bark_value = audio.bark.from_freq(500.0)
	assert np.isscalar(bark_value)
	freq_back = audio.bark.to_freq(bark_value)
	assert np.isscalar(freq_back)
	testing.assert_allclose(freq_back, 500.0, rtol=1e-3)


def test_get_bw_is_primary_and_calc_bw_alias_matches():
	fc = np.array([500.0, 1000.0])
	testing.assert_allclose(
		audio.bark.get_bw(fc),
		25 + 75 * (1 + 1.4 * (fc / 1000.0) ** 2) ** 0.69,
	)
	testing.assert_allclose(audio.bark.get_bw(fc), audio.bark.calc_bw(fc))
	testing.assert_allclose(audio.erb.get_bw(fc), 24.7 * (4.37 * (fc / 1000.0) + 1))
	testing.assert_allclose(audio.erb.get_bw(fc), audio.erb.calc_bw(fc))


def test_octave_bandwidth_matches_edges():
	fc = 1000.0
	bw = audio.octave.get_bw(fc, oct_fraction=3, base_system=2)
	ratio = 2 ** (1 / 3)
	upper = fc * np.sqrt(ratio)
	lower = fc / np.sqrt(ratio)
	testing.assert_allclose(bw, upper - lower)


def test_list_inputs_supported_across_scales():
	testing.assert_allclose(audio.bark.from_freq([100.0, 200.0]), [0.95573786, 1.9595463])
	testing.assert_allclose(audio.erb.from_freq([100.0, 200.0]), audio.erb.from_freq(np.array([100.0, 200.0])))
	testing.assert_allclose(audio.octave.from_freq([100.0, 200.0]), audio.octave.from_freq(np.array([100.0, 200.0])))


def test_octave_fraction_validation():
	with pytest.raises(ValueError):
		audio.octave.from_freq(1000.0, oct_fraction=0)
	with pytest.raises(ValueError):
		audio.octave.to_freq(30.0, oct_fraction=0)
	with pytest.raises(ValueError):
		audio.octave.get_bw(1000.0, oct_fraction=0)


def test_bark_raises_valueerror_for_out_of_range_frequency():
	with pytest.raises(ValueError):
		audio.bark.from_freq(10.0)
	with pytest.raises(ValueError):
		audio.bark.from_freq(16000.0)


def test_bark_limits_return_copy():
	limits = audio.bark.get_bark_limits()
	limits.append(99999)
	assert 99999 not in audio.bark.get_bark_limits()
