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

import warnings
warnings.filterwarnings("ignore", "Module (.*) was already import from")
warnings.filterwarnings("ignore", "The oldnumeric module",
                        DeprecationWarning)

# set metadata
from . import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

# register new units at the top level in astropy
try:
    from astropy import (__version__ as astropyversion, units)
except ImportError:
    pass
else:
    from distutils.version import LooseVersion
    if LooseVersion(astropyversion) < LooseVersion('0.3'):
        units.def_unit(['counts'], represents=units.Unit('count'),
                       register=True)
        units.def_unit(['strain'], represents=units.Unit(''), register=True)
    else:
        _counts = units.def_unit(['counts'], units.Unit('count'))
        _strain = units.def_unit(['strain'], units.Unit(''))
        units.add_enabled_units([_counts, _strain])

# ignore Quantity conversions in astropy 0.2
try:
    from astropy.units.quantity import WARN_IMPLICIT_NUMERIC_CONVERSION
except ImportError:
    pass
else:
    WARN_IMPLICIT_NUMERIC_CONVERSION.set(False)
