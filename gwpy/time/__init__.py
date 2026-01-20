# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""GPS time conversion utilities.

The :class:`~astropy.time.Time` object from the astropy package
is imported for user convenience, and a GPS time conversion function
is provided.

All other time conversions can be easily completed using the
:class:`~astropy.time.Time` object.
"""

from typing import TYPE_CHECKING

from astropy.time import Time

from ._ligotimegps import (
    LIGOTimeGPS,
    LIGOTimeGPSLike,
)
from ._tconvert import (
    from_gps,
    tconvert,
    to_gps,
)

if TYPE_CHECKING:
    from ._tconvert import SupportsToGps

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "LIGOTimeGPS",
    "LIGOTimeGPSLike",
    "from_gps",
    "tconvert",
    "to_gps",
]
