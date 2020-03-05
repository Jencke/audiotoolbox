import numpy as np
import audiotools as audio

class FrequencyDomainSignal(object):
    def __init__(self):
        self.waveform = np.array([], dtype=np.complex128)
        self.__fs = None

    @property
    def fs(self):
        """Get the signals sampling rate"""

        return self.__fs
    @fs.setter
    def fs(self, fs):
        """Set the signals sampling rate"""

        # If no fs provided or allready defined:
        if fs == None and self.__fs == None:
            raise ValueError('No sampling rate provided')

        # If fs is defined
        elif fs != None:
            if self.__fs == None:
                self.__fs = fs
            elif self.__fs != fs:
                raise ValueError('Sampling rate can\'t be changed')

    @property
    def real(self):
        return self.wavefrom.real

    @property
    def imag(self):
        return self.wavefrom.real

    @property
    def phase(self):
        return np.angle(self.waveform)

    angle = phase
