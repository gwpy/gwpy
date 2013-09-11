# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Package to do gravitational wave astrophysics with python
"""

# import core astropy modules
from astropy import table
try:
    from astropy.units.quantity import WARN_IMPLICIT_NUMERIC_CONVERSION
except ImportError:
    pass
else:
    WARN_IMPLICIT_NUMERIC_CONVERSION.set(False)

import warnings
warnings.filterwarnings("ignore", "Module (.*) was already import from")

# set metadata
from . import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

# register new unit at the top level
from astropy import units
units.def_unit(['counts'], represents=units.Unit('count'), register=True)
