# import pytest
# import numpy as np
# from audiotoolbox.filter.exponential_filter import design_efilt, apply_efilt, efilt


# # Mocking the audio.Signal class since we don't need the full implementation for testing logic
# class MockSignal:
#     def __init__(self, data, fs):
#         self.data = np.array(data)
#         self.fs = fs

#     # Allow numpy operations to work on the object or return the data
#     def __array__(self):
#         return self.data


# def test_apply_efilt_real_vs_complex():
#     fs = 100.0
#     t = np.arange(100) / fs
#     # Create simple sine wave
#     input_signal = np.sin(2 * np.pi * 10.0 * t)

#     # Filter design
#     b, a = design_efilt(10.0, 5.0, fs)

#     # Test complex return
#     out_complex = apply_efilt(input_signal, b, a, return_complex=True)
#     assert np.iscomplexobj(out_c.ndarrayomplex)

#     # Test real return
#     out_real = apply_efilt(input_signal, b, a, return_complex=False)
#     assert np.isrealobj(out_real)

#     # Real output should match real part of complex output
#     np.testing.assert_allclose(out_real, np.real(out_complex))


# def test_efilt_integration():
#     """Test the high-level wrapper function."""
#     fs = 1000.0
#     duration = 1.0
#     t = np.arange(int(fs * duration)) / fs
#     signal_data = np.random.randn(len(t))

#     # Mock signal object
#     sig_obj = MockSignal(signal_data, fs)

#     # Run efilt
#     result = efilt(sig_obj, fc=100.0, bw=20.0, return_complex=False)

#     assert len(result) == len(signal_data)
#     assert np.isrealobj(result)


# def test_peak_gain_at_center_frequency():
#     """Verify that the filter has ~unity gain at center frequency."""
#     fs = 10000.0
#     fc = 1000.0
#     bw = 500.0
#     duration = 1.0

#     t = np.arange(int(fs * duration)) / fs
#     # Generate a complex exponential at exactly fc
#     # This matches the rotation of the filter, so we expect steady state amplitude
#     input_signal = np.exp(1j * 2 * np.pi * fc * t)

#     b, a = design_efilt(fc, bw, fs)
#     output = apply_efilt(input_signal, b, a, return_complex=True)

#     # Ignore initial transient (first 1000 samples)
#     amplitudes = np.abs(output[1000:])

#     # The filter design normalizes the peak to 1.0
#     mean_amp = np.mean(amplitudes)
#     assert 0.95 < mean_amp < 1.05


# def test_attenuation_off_center():
#     """Verify that frequencies away from fc are attenuated."""
#     fs = 10000.0
#     fc = 1000.0
#     bw = 100.0  # Narrow bandwidth

#     # Input at 2000 Hz (far from 1000 Hz)
#     t = np.arange(int(fs)) / fs
#     input_signal = np.exp(1j * 2 * np.pi * 2000.0 * t)

#     b, a = design_efilt(fc, bw, fs)
#     output = apply_efilt(input_signal, b, a, return_complex=True)

#     # Should be significantly attenuated
#     mean_amp = np.mean(np.abs(output[1000:]))
#     assert mean_amp < 0.2


# # filepath: /home/joerg/Projects/repos/audiotoolbox/tests/test_exponential_filter.py
# import pytest
# import numpy as np
# from audiotoolbox.filter.exponential_filter import design_efilt, apply_efilt, efilt


# # Mocking the audio.Signal class since we don't need the full implementation for testing logic
# class MockSignal:
#     def __init__(self, data, fs):
#         self.data = np.array(data)
#         self.fs = fs

#     # Allow numpy operations to work on the object or return the data
#     def __array__(self):
#         return self.data


# def test_design_efilt_coefficients():
#     fc = 1000.0
#     bw = 100.0
#     fs = 44100.0

#     b, a = design_efilt(fc, bw, fs)

#     # Check dimensions
#     assert len(b) == 1
#     assert len(a) == 2

#     # Check that coefficients are complex
#     assert np.iscomplexobj(a)

#     # Check that a[0] is 1 (standard form)
#     assert a[0] == 1.0


# def test_apply_efilt_real_vs_complex():
#     fs = 100.0
#     t = np.arange(100) / fs
#     # Create simple sine wave
#     input_signal = np.sin(2 * np.pi * 10.0 * t)

#     # Filter design
#     b, a = design_efilt(10.0, 5.0, fs)

#     # Test complex return
#     out_complex = apply_efilt(input_signal, b, a, return_complex=True)
#     assert np.iscomplexobj(out_complex)

#     # Test real return
#     out_real = apply_efilt(input_signal, b, a, return_complex=False)
#     assert np.isrealobj(out_real)

#     # Real output should match real part of complex output
#     np.testing.assert_allclose(out_real, np.real(out_complex))


# def test_efilt_integration():
#     """Test the high-level wrapper function."""
#     fs = 1000.0
#     duration = 1.0
#     t = np.arange(int(fs * duration)) / fs
#     signal_data = np.random.randn(len(t))

#     # Mock signal object
#     sig_obj = MockSignal(signal_data, fs)

#     # Run efilt
#     result = efilt(sig_obj, fc=100.0, bw=20.0, return_complex=False)

#     assert len(result) == len(signal_data)
#     assert np.isrealobj(result)


# def test_peak_gain_at_center_frequency():
#     """Verify that the filter has ~unity gain at center frequency."""
#     fs = 10000.0
#     fc = 1000.0
#     bw = 500.0
#     duration = 1.0

#     t = np.arange(int(fs * duration)) / fs
#     # Generate a complex exponential at exactly fc
#     # This matches the rotation of the filter, so we expect steady state amplitude
#     input_signal = np.exp(1j * 2 * np.pi * fc * t)

#     b, a = design_efilt(fc, bw, fs)
#     output = apply_efilt(input_signal, b, a, return_complex=True)

#     # Ignore initial transient (first 1000 samples)
#     amplitudes = np.abs(output[1000:])

#     # The filter design normalizes the peak to 1.0
#     mean_amp = np.mean(amplitudes)
#     assert 0.95 < mean_amp < 1.05


# def test_attenuation_off_center():
#     """Verify that frequencies away from fc are attenuated."""
#     fs = 10000.0
#     fc = 1000.0
#     bw = 100.0  # Narrow bandwidth

#     # Input at 2000 Hz (far from 1000 Hz)
#     t = np.arange(int(fs)) / fs
#     input_signal = np.exp(1j * 2 * np.pi * 2000.0 * t)

#     b, a = design_efilt(fc, bw, fs)
#     output = apply_efilt(input_signal, b, a, return_complex=True)

#     # Should be significantly attenuated
#     mean_amp = np.mean(np.abs(output[1000:]))
#     assert mean_amp < 0.2
