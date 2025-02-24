# Copyright (C) Cardiff University (2024-)
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

"""Type annotation tools for GWpy."""

from __future__ import annotations

import typing

try:
    from typing import Self
except ImportError:  # python < 3.11
    from typing_extensions import Self

from astropy.units import UnitBase
from numpy.typing import (
    ArrayLike,
    DTypeLike,
)

try:
    from astropy.units.typing import QuantityLike
except ImportError:  # astropy < 6.1
    from astropy.units import Quantity as _Quantity
    QuantityLike = typing.Union[_Quantity, ArrayLike]


from .time import (
    GpsConvertible,
    GpsType,
)

# Gps types
GpsLike = typing.Union[GpsType, GpsConvertible]

# Unit types
UnitLike = typing.Union[UnitBase, str, None]
