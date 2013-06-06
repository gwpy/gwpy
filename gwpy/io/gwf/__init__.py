# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Read GWF files into arrays
"""

from astropy import units

from ... import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

# register unit alias
units.def_unit(['counts'], represents=units.Unit('count'), register=True)
