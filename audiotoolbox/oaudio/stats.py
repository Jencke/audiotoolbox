from .. import audiotools as audio
import numpy as np


class BaseStats(object):
    """Class containing"""

    def __init__(self, sig):
        self.sig = sig

    @property
    def mean(self):
        """aritmetic mean"""
        mean = np.mean(self.sig, axis=0)
        return mean

    @property
    def var(self):
        """variance"""
        return np.var(self.sig, axis=0)


class SignalStats(BaseStats):
    def __init__(self, sig):
        BaseStats.__init__(self, sig)

    @property
    def dbspl(self):
        """Soundpressure level relative to 20uPa in dB

        See Also
        --------
        audiotools.calc_dbspl
        """
        return audio.calc_dbspl(self.sig)

    @property
    def dbfs(self):
        """Level in dB full scale

        See Also
        --------
        audiotools.calc_dbfs
        """
        return audio.calc_dbfs(self.sig)

    @property
    def crest_factor(self):
        """Soundpressure level relative to 20uPa in dB

        See Also
        --------
        audiotools.crest_factor
        """
        return audio.crest_factor(self.sig)

    @property
    def dba(self):
        """A weighted sound pressure level in dB


        See Also
        --------
        audiotools.filter.a_weighting
        """
        a_weighted = audio.filter.a_weighting(self.sig)
        return a_weighted.stats.dbspl

    @property
    def dbc(self):
        """A weighted sound pressure level in dB


        See Also
        --------
        audiotools.filter.a_weighting
        """
        c_weighted = audio.filter.c_weighting(self.sig)
        return c_weighted.stats.dbspl


class FreqDomainStats(BaseStats):
    def __init__(self, sig):
        BaseStats.__init__(self, sig)
