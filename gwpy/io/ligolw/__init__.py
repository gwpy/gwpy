# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Read LIGO_LW-format XML into Tables
"""

from ... import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

#from .connect import read_ligolw as read
#from .segments import read_ligolw_segments as read_segments

__all__ = ['read', 'read_segments']
