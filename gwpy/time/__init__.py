# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-)
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

The :class:`~astropy.time.Time` object from the astropy package
is imported for user convenience, and a GPS time conversion function
is provided.

All other time conversions can be easily completed using the `Time`
object.
"""

from importlib import import_module

from astropy.time import Time

from ._ligotimegps import (
    GPS_TYPES,
    GpsType,
    LIGOTimeGPS,
)
from ._tconvert import (
    from_gps,
    tconvert,
    to_gps,
)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
