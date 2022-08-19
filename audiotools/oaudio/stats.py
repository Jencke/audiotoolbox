from .. import audiotools as audio
import numpy as np

class BaseStats(object):
    """Class containing """

    def __init__(self, obj):
        self.sig = obj

    def mean(self):
        '''aritmetic mean'''
        mean = np.mean(self.sig, axis=0)
        return mean

    def var(self):
        '''variance'''
        return np.var(self.sig, axis=0)

    def dbspl(self):
        '''Soundpressure level relative to 20uPa in dB'''
        return audio.calc_dbspl(self.sig)

    def crest_factor(self):
        '''Soundpressure level relative to 20uPa in dB'''
        return audio.crest_factor(self.sig)

# def SignalStats(BaseStats):
#     def __init__(self, obj)
