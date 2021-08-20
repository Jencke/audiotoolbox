from sys import version_info, exit
if not (version_info.major == 3
        and version_info.minor >= 0):  # pragma: no cover
    print("Audiotools requires a python version > 3.0")
    print("You are currently using Python {}.{}.".format(version_info.major, version_info.minor))
    exit(1)

from .audiotools import *
from . oaudio import Signal, as_signal
from . oaudio import FrequencyDomainSignal
from . import interfaces
