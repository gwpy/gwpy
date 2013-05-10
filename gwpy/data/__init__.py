# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Module providing classes for reading, writing and manipulating
time-series and spectrum data.
"""

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from .series import *

# import IO routines
from ..io.gwf import connect

__all__ = ["NDData", "TimeSeries", "Spectrum", "Spectrogram"]

