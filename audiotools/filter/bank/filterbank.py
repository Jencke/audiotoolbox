from typing import Literal
from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike

from .. import gammatone_filt as gamma, butterworth_filt as butter
from .. import brickwall_filt as brick
from ... import audiotools as audio


class FilterBank(object):
    '''Parent Class for all filterbanks

    Parameters
    ----------
    fc : ndarray
        Center frequencies in Hz
    bw : ndarray
        Filter Bandwidths in Hz
    fs : int
        Sampling frequency
    **kwargs :
        Further paramters such as filter order to pass to the Filter
        function, see filter documenation for details. Value can
        either be an ndarray that matches the length of `fc` or a
        single value in which case this value is used for all filters.

    '''
    def __init__(self, fc, bw, fs, **kwargs):
        self.fc = np.asarray([fc]).flatten()
        self.bw = np.asarray([bw]).flatten()
        self.fs = fs
        self.params = dict()

        if self.fc.shape != self.bw.shape:
            raise Exception(
                'Length of center frequencies must equal length of bandwidths')

        self._update_params(**kwargs)

    @property
    def n_filters(self):
        return len(self.fc)

    def _update_params(self, **kwargs):
        ''' Used to update the parameter dict
        '''
        n_val = self.n_filters
        for k, v in kwargs.items():
            if np.ndim(v):
                if len(v) == n_val:
                    self.params[k] = np.asarray(v)
                else:
                    raise Exception(f'Size missmatch in parameter \'{k}\'')
            else:
                self.params[k] = np.asarray(n_val * [v])

    def __len__(self):
        return self.n_filters

    def __getitem__(self, i):
        bank = deepcopy(self)
        bank.bw = self.bw[i]
        bank.fc = self.fc[i]
        bank.fs = self.fs
        for k, v in self.params.items():
            bank.params[k] = v[i]
        return bank


class ButterworthBank(FilterBank):
    def __init__(self, fc, bw, fs, **kwargs):
        super().__init__(fc, bw, fs, **kwargs)

        # set default parameters
        if 'order' not in self.params.keys():
            self._update_params(order=2)

        # Calculate filter coefficents
        self.coefficents = np.zeros((np.max(self.params['order']), 6,
                                     self.n_filters))
        for i_filt in range(self.n_filters):
            # extract parameter set for current filter
            current_params = {k: v[i_filt] for k, v in self.params.items()}
            order = current_params['order']
            low_f = self.fc[i_filt] - self.bw[i_filt] / 2
            high_f = self.fc[i_filt] + self.bw[i_filt] / 2
            # design filter and save coefficents
            sos = butter.design_butterworth(low_f, high_f, self.fs,
                                            **current_params)
            self.coefficents[:order, :, i_filt] = sos

    def filt(self, signal):
        n_ch_out = (*signal.shape[1:], self.n_filters)
        duration = len(signal) / self.fs
        out_sig = audio.Signal(n_ch_out, duration, self.fs)
        for i_filt, freq in enumerate(self.fc):
            order = self.params['order'][i_filt]
            # sos has to be C-contigous
            sos = self.coefficents[:order, :, i_filt].copy(order='C')
            out, states = butter.apply_sos(signal, sos)
            out_sig.T[i_filt] = out.T
        return out_sig

    def __getitem__(self, i):
        bank = super().__getitem__(i)
        bank.coefficents = self.coefficents[:, :, i]
        return bank


class GammaToneBank(FilterBank):
    def __init__(self, fc, bw, fs, **kwargs):
        FilterBank.__init__(self, fc, bw, fs, **kwargs)

        self._update_params(**kwargs)
        # set default parameters
        if 'order' not in self.params.keys():
            self._update_params(order=4)
        if 'attenuation_db' not in self.params.keys():
            self._update_params(attenuation_db='erb')

        # Calculate filter coefficents
        self.coefficents = np.zeros([4, self.n_filters], complex)

        for i_filt in range(self.n_filters):
            current_params = {k: v[i_filt]
                              for k, v in self.params.items()}

            b, a = gamma.design_gammatone(self.fc[i_filt],
                                          self.bw[i_filt], self.fs,
                                          **current_params)
            self.coefficents[0, i_filt] = b[0]
            self.coefficents[2:, i_filt] = a

    def filt(self, signal):
        n_ch_out = (*signal.shape[1:], self.n_filters)
        duration = len(signal) / self.fs
        out_sig = audio.Signal(n_ch_out, duration, self.fs, dtype=complex)
        for i_filt, freq in enumerate(self.fc):
            order = self.params['order'][i_filt]
            coeff = self.coefficents[:, i_filt]
            b = coeff[0],
            a = coeff[2:]
            out, states = gamma.gammatonefos_apply(signal, b, a, order)
            out_sig.T[i_filt] = out.T

        # squeeze to leave dimensions unchanged if n_filters == 1
        if out_sig.shape[-1] == 1:
            out_sig = out_sig.squeeze(-1)
        return out_sig

    def __getitem__(self, i):
        bank = super().__getitem__(i)
        bank.coefficents = self.coefficents[:, i]
        return bank


class BrickBank(FilterBank):
    def __init__(self, fc, bw, fs):
        FilterBank.__init__(self, fc, bw, fs)

    def filt(self, signal):
        n_ch_out = (*signal.shape[1:], self.n_filters)
        duration = len(signal) / self.fs
        out_sig = audio.Signal(n_ch_out, duration, self.fs)
        for i_filt, (freq, bw) in enumerate(zip(self.fc, self.bw)):
            low_f = freq - bw / 2
            high_f = freq + bw / 2
            out = brick.brickwall(signal, low_f, high_f, self.fs)
            out_sig.T[i_filt] = out.T
        return out_sig


def create_filterbank(fc: ArrayLike,
                      bw: ArrayLike,
                      filter_type: Literal['butter', 'gammatone', 'brickwall'],
                      fs: int,
                      **kwargs) -> FilterBank:
    '''Creates a filterbank object

    Parameters
    ----------
    fc : ndarray
        Center frequencies in Hz
    bw : ndarray
        Filter Bandwidths in Hz
    filter_type : 'butter', 'gammatone'
        Type of Bandpass filter
    fs : int
        Sampling frequency
    **kwargs
        Further paramters such as filter order to pass to the Filter
        function, see filter documenation for details. Value can either be
        an ndarray that matches the length of `fc` or a single value in
        which case this value is used for all filters.

    Returns
    -------
       FilterBank Object

    '''

    if filter_type == 'butter':
        bank = ButterworthBank(fc, bw, fs, **kwargs)
    elif filter_type == 'gammatone':
        bank = GammaToneBank(fc, bw, fs, **kwargs)
    elif filter_type == 'brickwall':
        bank = BrickBank(fc, bw, fs)
    return bank
