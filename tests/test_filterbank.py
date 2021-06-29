import unittest
import numpy as np

import audiotools as audio
import audiotools.filter.filterbank as bank
from audiotools.filter import gammatone_filt
from audiotools.filter import butterworth_filt
import numpy.testing as testing


class test_oaudio(unittest.TestCase):
    def test_create_filterbank(self):
        fc = [100, 200]
        bw = [10, 20]
        butter = audio.create_filterbank(fc, bw, 'butter', 48000)
        assert isinstance(butter, bank.ButterworthBank)

        gamma = audio.create_filterbank(fc, bw, 'gammatone', 48000)
        assert isinstance(gamma, bank.GammaToneBank)

    def test_butterworth(self):
        fc = [100, 200, 5000]
        bw = [10, 5, 8]
        fs = 48000
        butter = audio.create_filterbank(fc, bw, 'butter', fs)

        for i_filt, (fc, bw) in enumerate(zip(fc, bw)):
            low_f = fc - bw / 2
            high_f = fc + bw / 2
            sos = butterworth_filt.design_butterworth(low_f, high_f, fs)
            coeff = butter.coefficents[:, :, i_filt]
            testing.assert_array_equal(sos, coeff)

        # Test filter gain
        sig = audio.Signal(1, 1, 48000)
        fc = np.round(audio.freqarange(100, 4000, 1, 'erb'))        
        bw = audio.calc_bandwidth(fc, 'erb')
        # add a tone at every fc
        for f in fc:
            sig.add_tone(f)
        # create filterbank an run signal
        bank = audio.create_filterbank(fc, bw, 'butter', 48000)
        sig_out = bank.filt(sig)
        #check amplitudes at fc of every filter
        amps = np.zeros(len(fc))
        for i_fc, f in enumerate(fc):
            spec = sig_out.ch[i_fc].to_freqdomain()
            amps[i_fc] = np.abs(spec[spec.freq==f])
        # Amplitudes should be 0.5 (two sided spectrum)
        assert np.all((amps - 0.5) <= 0.01)
        

    def test_gammatone(self):
        # Test if filter coefficents are equivalent tho those of
        # individual filters
        fc = [100, 200, 5000]
        bw = [10, 5, 8]
        fs = 48000
        gamma = audio.create_filterbank(fc, bw, 'gammatone', fs)
        for i_filt, (fc, bw) in enumerate(zip(fc, bw)):
            b, a = gammatone_filt.design_gammatone(fc, bw, fs)
            b_bank = gamma.coefficents[0, i_filt]
            a_bank = gamma.coefficents[2:, i_filt]            
            testing.assert_array_equal(a, a_bank)
            testing.assert_array_equal(b, b_bank)

        # Test the gain at fc 
        sig = audio.Signal(1, 1, 48000)
        fc = np.round(audio.freqarange(100, 4000, 1, 'erb'))        
        bw = audio.calc_bandwidth(fc, 'erb')
        # add a tone at every fc
        for f in fc:
            sig.add_tone(f)
        # create filterbank an run signal
        bank = audio.create_filterbank(fc, bw, 'gammatone', 48000)
        sig_out = bank.filt(sig)
        #check amplitudes at fc of every filter
        amps = np.zeros(len(fc))
        for i_fc, f in enumerate(fc):
            spec = sig_out.ch[i_fc].to_freqdomain()
            amps[i_fc] = np.abs(spec[spec.freq==f])
        # Amplitudes hould be 1 (one sided spectrum)
        assert np.all((amps - 1) <= 0.01)


    def test_brickwall(self):
        # Test the gain at fc 
        sig = audio.Signal(1, 1, 48000)
        fc = np.round(audio.freqarange(100, 4000, 1, 'erb'))        
        bw = audio.calc_bandwidth(fc, 'erb')
        # add a tone at every fc
        for f in fc:
            sig.add_tone(f)
        # create filterbank an run signal
        bank = audio.create_filterbank(fc, bw, 'brickwall', 48000)
        sig_out = bank.filt(sig)
        #check amplitudes at fc of every filter
        amps = np.zeros(len(fc))
        for i_fc, f in enumerate(fc):
            spec = sig_out.ch[i_fc].to_freqdomain()
            amps[i_fc] = np.abs(spec[spec.freq==f])
        # Amplitudes hould be 0.5 (double sided spectrum)
        assert np.all((amps - 0.5) <= 0.01)
        

    def test_set_params(self):
        fc = [100, 200, 5000]
        bw = [10, 5, 8]
        fs = 48000
        print()
        gamma = audio.create_filterbank(fc, bw, 'gammatone', fs,
                                        order=5, attenuation_db=-3)
        print('--')

        for i_filt, (fc, bw) in enumerate(zip(fc, bw)):
            b, a = gammatone_filt.design_gammatone(fc, bw, fs, order=5,
                                                   attenuation_db=-3)
            b_bank = gamma.coefficents[0, i_filt]
            a_bank = gamma.coefficents[2:, i_filt]            
            testing.assert_array_equal(a, a_bank)
            testing.assert_array_equal(b, b_bank)

