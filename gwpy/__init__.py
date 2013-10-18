# Copyright (C) Duncan Macleod (2013)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.


"""Package to do gravitational wave astrophysics with python
"""

# import core astropy modules
from astropy import table
try:
# from astropy.units.quantity import WARN_IMPLICIT_NUMERIC_CONVERSION
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
units.def_unit(['strain'], represents=units.Unit(''), register=True)
