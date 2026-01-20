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

"""LIGOTimeGPS object discovery."""

# ruff: noqa: I001

from __future__ import annotations

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

from typing import (
    TYPE_CHECKING,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from typing import Self

# try and import LIGOTimeGPS from LAL, otherwise use the pure-python backup
# provided by the ligotimegps package, its slower, but works
try:
    # import h5py first to avoid https://git.ligo.org/lscsoft/lalsuite/-/issues/821
    import h5py  # noqa: F401
    # then import LAL
    from lal import LIGOTimeGPS
except ImportError:
    from ligotimegps import LIGOTimeGPS

__all__ = [
    "LIGOTimeGPS",
    "LIGOTimeGPSLike",
]


@runtime_checkable
class LIGOTimeGPSLike(Protocol):
    """Protocol for types that are implementations of LIGOTimeGPS.

    This is used for type hinting functions that can accept
    |lal.LIGOTimeGPS|_, or `ligotimegps.LIGOTimeGPS`, or any
    other implementation of the ``LIGOTimeGPS`` standard.

    This can also be used at runtime with `isinstance` to check if an
    object is a LIGOTimeGPS-like object, but cannot be used with
    `issubclass`.
    """

    gpsSeconds: int      # noqa: N815
    gpsNanoSeconds: int  # noqa: N815

    def __init__(self, new: float | LIGOTimeGPS | str) -> None: ...

    def __float__(self) -> float: ...

    def __add__(self, other: Self | float) -> Self: ...
    def __radd__(self, other: Self | float) -> Self: ...
