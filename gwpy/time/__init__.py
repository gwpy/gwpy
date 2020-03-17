# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""This module provides time conversion utilities.

The :class:`~astropy.time.core.Time` object from the astropy package
is imported for user convenience, and a GPS time conversion function
is provided.

All other time conversions can be easily completed using the `Time`
object.
"""

from importlib import import_module

from astropy.time import Time

# try and import LIGOTimeGPS from LAL, otherwise use the pure-python backup
# provided by the ligotimegps package, its slower, but works
try:
    from lal import LIGOTimeGPS
except ImportError:
    from ligotimegps import LIGOTimeGPS

from ._tconvert import (tconvert, to_gps, from_gps)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# build list of compatible gps types
gps_types = []
for _modname in ('lal', 'ligotimegps', 'glue.lal',):
    try:
        _mod = import_module(_modname)
    except ImportError:  # library not installed
        continue
    try:
        gps_types.append(getattr(_mod, 'LIGOTimeGPS'))
    except AttributeError:  # no LIGOTimeGPS available
        continue
gps_types = tuple(gps_types)

del import_module
