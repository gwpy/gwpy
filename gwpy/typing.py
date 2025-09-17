# Copyright (c) 2024-2025 Cardiff University
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

from typing import (
    TypeAlias,
)

try:
    from typing import Self
except ImportError:  # python < 3.11
    from typing_extensions import Self

from astropy.units import UnitBase
from astropy.units.typing import QuantityLike
from numpy.typing import (
    ArrayLike,
    DTypeLike,
)

from .time import (
    GpsConvertible,
    GpsType,
)

# Gps types
GpsLike: TypeAlias = GpsType | GpsConvertible

# Unit types
UnitLike: TypeAlias = UnitBase | str | None
