# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Test availability of each of the LALSuite SWIG bindings
"""

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ["SWIG_LAL"]

# test presence of SWIG-wrappings for LAL
try:
    from lal import lal as swiglal
except ImportError:
    SWIG_LAL = False
else:
    SWIG_LAL = True
    __all__.append('swiglal')
