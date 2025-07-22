from .. import audiotoolbox as audio
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

    @property
    def rms(self):
        r"""Root mean square.

        Returns
        -------
        float : The RMS value
        """
        rms = np.sqrt(np.mean(self.sig**2, axis=0))
        return rms


class SignalStats(BaseStats):
    def __init__(self, sig):
        BaseStats.__init__(self, sig)

    @property
    def dbspl(self):
        """Soundpressure level relative to 20uPa in dB

        See Also
        --------
        audiotoolbox.calc_dbspl
        """
        return audio.calc_dbspl(self.sig)

    @property
    def dbfs(self) -> np.ndarray:
        """Level in dB full scale

        See Also
        --------
        audiotoolbox.calc_dbfs
        """
        return audio.calc_dbfs(self.sig)

    @property
    def crest_factor(self):
        """Soundpressure level relative to 20uPa in dB

        See Also
        --------
        audiotoolbox.crest_factor
        """
        return audio.crest_factor(self.sig)

    @property
    def dba(self):
        """A weighted sound pressure level in dB


        See Also
        --------
        audiotoolbox.filter.a_weighting
        """
        a_weighted = audio.filter.a_weighting(self.sig)
        return a_weighted.stats.dbspl

    @property
    def dbc(self):
        """A weighted sound pressure level in dB


        See Also
        --------
        audiotoolbox.filter.a_weighting
        """
        c_weighted = audio.filter.c_weighting(self.sig)
        return c_weighted.stats.dbspl

    def octave_band_levels(
        self, oct_fraction: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate octave band levels of the signal.

        Parameters
        ----------
        oct_fraction : float, optional
            Fraction of an octave to use, by default 3 for 1/3 octave bands.

        Returns
        -------
        tuple : (frequencies, levels)
            Frequencies and corresponding levels in dB Full Scale (dBFS)

        See Also
        --------
        audiotoolbox.filter.bank.octave_bank
        """
        bank = audio.filter.bank.octave_bank(self.sig.fs, oct_fraction=oct_fraction)
        bank_out = bank.filt(self.sig.ch[0])
        return bank.fc, bank_out.stats.dbfs


class FreqDomainStats(BaseStats):
    def __init__(self, sig):
        BaseStats.__init__(self, sig)
