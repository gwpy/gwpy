# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines some conveniences mapping between types and
functions from the LIGO Algorithm Library and AstroPy
"""

from ... import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from .swig import *

if SWIG_LAL:
    #from lal import lal as swiglal
    from .atomic import *

__all__ = ['SWIG_LAL']
