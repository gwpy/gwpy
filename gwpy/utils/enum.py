# Copyright (c) 2019-2025 Cardiff University
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

"""Utilties for enumerations."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    import builtins
    from typing import Self

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class NumpyTypeEnum(Enum):
    """`~enum.Enum` of numpy types."""

    @property
    def dtype(self) -> numpy.dtype:
        """The numpy dtype corresponding to this enumerated type."""
        return numpy.dtype(self.name.lower())

    @property
    def type(self) -> builtins.type:
        """The python type corresponding to this enumerated type."""
        return self.dtype.type

    @classmethod
    def find(
        cls,
        type_: builtins.type | str | int,
    ) -> Self:
        """Return the enumerated type corresponding to the given python type."""
        try:
            return cls(type_)
        except ValueError as exc:
            if isinstance(type_, str):
                type_ = type_.lower()
            try:
                return cls[numpy.dtype(type_).name.upper()]  # type: ignore[arg-type]
            except (
                KeyError,  # numpy dtype isn't support by this enum
                TypeError,  # type isn't a valid numpy type
            ):
                raise exc from None
