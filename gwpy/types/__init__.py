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

"""The core `Array` object and direct subclasses.

These objects form the basic 1-D and 2-D arrays with metadata from which
we can build specific data representations like the `TimeSeries`.
"""

from .array import Array
from .array2d import Array2D
from .index import Index
from .series import Series

from . import io

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = [
    "Array",
    "Array2D",
    "Index",
    "Series",
]
