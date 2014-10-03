# -*- coding: utf-8 -*-
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

"""This module provides the core `Array` object and direct subclasses.

These objects form the basic 1-D and 2-D arrays with metadata from which
we can build specific data representations like the `TimeSeries`.
"""

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from glue.lal import (Cache, CacheEntry)

from .array import *
from .array2d import *
from .series import *

# define custom time-series units
try:
    from astropy import (__version__ as astropyversion, units)
except ImportError:
    pass
else:
    from distutils.version import LooseVersion
    if LooseVersion(astropyversion) < LooseVersion('0.3'):
        units.def_unit(['counts'], represents=units.Unit('count'),
                       register=True)
        units.def_unit(['strain'], represents=units.dimensionless_unscaled,
                       register=True)
        units.def_unit(['coherence'], represents=units.dimensionless_unscaled,
                       register=True)
    else:
        units.add_enabled_units([
            units.def_unit(['counts'], units.Unit('count')),
            units.def_unit(['coherence'], units.dimensionless_unscaled),
            units.def_unit(['strain'], units.dimensionless_unscaled),
            ])

__all__ = ['Array', 'Array2D', 'Series', 'Cache', 'CacheEntry']

