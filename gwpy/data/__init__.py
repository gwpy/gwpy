# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Module providing classes for reading, writing and manipulating
time-series and spectrum data.
"""

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from .nddata import *
from .array import *
from .array2d import *
from .series import *
from glue.lal import (Cache, CacheEntry)

__all__ = ['NDData', 'Array', 'Array2D', 'Series', 'Cache', 'CacheEntry']

