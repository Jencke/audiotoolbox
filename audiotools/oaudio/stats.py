from .. import audiotools as audio
import numpy as np

class BaseStats(object):
    """Class containing """

    def __init__(self, sig):
        self.sig = sig

    def mean(self):
        """aritmetic mean"""
        mean = np.mean(self.sig, axis=0)
        return mean

    def var(self):
        """variance"""
        return np.var(self.sig, axis=0)


class SignalStats(BaseStats):
    def __init__(self, sig):
        BaseStats.__init__(self, sig)

    def dbspl(self):
        """Soundpressure level relative to 20uPa in dB

        See Also
        --------
        audiotools.calc_dbspl
        """
        return audio.calc_dbspl(self.sig)

    def dbfs(self):
        """Level in dB full scale

        See Also
        --------
        audiotools.calc_dbfs
        """
        return audio.calc_dbfs(self.sig)

    def crest_factor(self):
        """Soundpressure level relative to 20uPa in dB

        See Also
        --------
        audiotools.crest_factor
        """
        return audio.crest_factor(self.sig)


class FreqDomainStats(BaseStats):
    def __init__(self, sig):
        BaseStats.__init__(self, sig)
