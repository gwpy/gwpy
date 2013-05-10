# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Extension of the astropy table module to include new I/O
"""

import warnings
warnings.filterwarnings('ignore', 'column name', UserWarning)

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from astropy.table import *

from ..io.ligolw import connect
