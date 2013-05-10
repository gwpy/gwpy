
"""Construct and manipulate time-domain window functions
"""

from .. import version

__author__ = "Duncan M. Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from numpy import kaiser, hamming, hanning

def kaiser_factory(beta):
    return lambda x: kaiser(len(x), beta) * x
