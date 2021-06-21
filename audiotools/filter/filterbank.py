import gammatone_filt as gamma
import butterworth_filt as butter
import numpy as np

def create_bank(fc, bw, filter_type, fs, **kwargs):
    if filter_type == 'butter':
        bank = ButterworthBank(fc, bw, fs, **kwargs)
    elif filter_type == 'gammatone':
        bank = GammaToneBank(fc, bw, fs, **kwargs)
    return bank


class FilterBank(object):
    def __init__(self, fc, bw, fs, **kwargs):
        self.cf = np.asarray(fc)
        self.bw = np.asarray(bw)
        self.fs = fs
        self.n_filters = len(self.cf)

        if self.cf.shape != self.bw.shape:
            raise Exception('Length of center frequencies must equal length of bandwidths')

def update_params(param, n_val, **kwargs):
    for k, v in kwargs.items():
        if k in param:
            if np.ndim(v):
                if len(v) == n_val:
                    param[k] = v
                else:
                    raise Exception(f'Size missmatch in parameter \'{k}\'')
            else:
                param[k] = n_val * [v]
    return param


class ButterworthBank(FilterBank):
    def __init__(self, fc, bw, fs, **kwargs):
        FilterBank.__init__(self, fc, bw, fs, **kwargs)

        # set default parameters
        self.params = {'order' : self.n_filters * [2]}
        # update defaults with predefined parameters
        self.params = update_params(self.params, self.n_filters, **kwargs)

        # Calculate filter coefficents
        self.coefficents = np.zeros((np.max(self.params['order']), 6,
                                     self.n_filters))

        for i_filt in range(self.n_filters):
            current_params = {k:v[i_filt] for k, v in self.params.items()}
            order = current_params['order']
            low_f = self.cf[i_filt] - self.bw[i_filt] / 2
            high_f = self.cf[i_filt] + self.bw[i_filt] / 2
            sos = butter.design_butterworth(low_f, high_f, self.fs, **current_params)
            self.coefficents[:order, :, i_filt] = sos

    def filt(self, signal):
        n_ch_out = (*signal.shape[1:], self.n_filters)
        duration = len(signal) / self.fs
        out_sig = audio.Signal(n_ch_out, duration, self.fs)
        for i_filt, freq in enumerate(self.cf):
            order = self.params['order'][i_filt]
            # sos has to be C-contigous
            sos = self.coefficents[:order, :, i_filt].copy(order='C')
            out, states = butter.apply_sos(signal, sos, states=True)
            out_sig.T[i_filt] = out.T
        return out_sig

class GammaToneBank(FilterBank):
    def __init__(self, fc, bw, fs, **kwargs):
        FilterBank.__init__(self, fc, bw, fs, **kwargs)

        # set default parameters
        self.params = {'order' : self.n_filters * [5],
                       'attenuation_db' : self.n_filters * ['erb']}
        # update defaults with predefined parameters
        self.params = update_params(self.params, self.n_filters, **kwargs)

        # Calculate filter coefficents
        self.coefficents = np.zeros([4, self.n_filters], complex)

        for i_filt in range(self.n_filters):
            current_params = {k:v[i_filt] for k, v in self.params.items()}
            order = current_params['order']
            # low_f = self.cf[i_filt] - self.bw[i_filt] / 2
            # high_f = self.cf[i_filt] + self.bw[i_filt] / 2
            b, a = gamma.design_gammatone(self.cf[i_filt],
                                          self.bw[i_filt], self.fs, **current_params)
            print(b)
            print(a)
            self.coefficents[0, i_filt] = b[0]
            self.coefficents[2:, i_filt]= a

            # self.coefficents[:order, :, i_filt] = sos

    def filt(self, signal):
        n_ch_out = (*signal.shape[1:], self.n_filters)
        duration = len(signal) / self.fs
        out_sig = audio.Signal(n_ch_out, duration, self.fs, dtype=complex)
        for i_filt, freq in enumerate(self.cf):
            order = self.params['order'][i_filt]
            coeff = self.coefficents[:, i_filt]
            b = coeff[0],
            a = coeff[2:]
            out, states = gamma.gammatonefos_apply(signal, b, a, order)
            out_sig.T[i_filt] = out.T
        return out_sig

import audiotools as audio
import matplotlib.pyplot as plt



# freqs = audio.freqarange(125, 4000, 1, 'erb')
# bws = audio.calc_bandwidth(freqs, 'erb')
freqs = [500, 2000]
bws = [100, 100]
#
sig = audio.Signal(1, 1, 48000).add_noise().add_fade_window(200e-3)
bank = create_bank(freqs, bws, 'gammatone', 48000)
sig_out = bank.filt(sig)

b, a = gamma.design_gammatone(500, 100, 48000)

fsig = sig_out.to_freqdomain()

fsigo = sig.to_freqdomain()
# fsigo2 = sig.copy().bandpass(500, 80, 'gammatone').to_freqdomain()
#
plt.plot(fsigo.freq, fsigo.abs())
plt.plot(fsig.freq, fsig[:, 0].abs())
plt.plot(fsig.freq, fsig[:, -1].abs())
plt.xlim(0, 4000)
# plt.plot(fsig.freq, fsig[:, 2].abs())
# plt.plot(fsig.freq, fsig[:, 3].abs())
# plt.plot(fsig.freq, fsig[:, 4].abs())
# plt.plot(fsig.freq, fsig[:, 5].abs())



#plt.plot(sig_out.T[0].real)
#plt.plot(sig_out.T[1].real)
#plt.plot(sig_out.T[2].real)
# plt.plot(sig_out.T[1].real)


#sig_out2 = filt_bank2.filt(sig)









# plt.plot(sig_out[:, 0, :])
