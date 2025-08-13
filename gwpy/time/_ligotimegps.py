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

"""LIGOTimeGPS object discovery."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# try and import LIGOTimeGPS from LAL, otherwise use the pure-python backup
# provided by the ligotimegps package, its slower, but works
try:
    from lal import LIGOTimeGPS
except ImportError:
    from ligotimegps import LIGOTimeGPS  # noqa: F401


def _import_ligotimegps(
    modname: str,
) -> type | None:
    """Return `True` if ``modname`` provides a usable ``LIGOTimeGPS``."""
    try:
        mod = import_module(modname)
    except ImportError:  # library not installed
        return None
    try:
        return mod.LIGOTimeGPS
    except AttributeError:  # no LIGOTimeGPS available
        return None


GPS_TYPES: tuple[type, ...] = tuple(filter(None, map(
    _import_ligotimegps,
    (
        "lal",
        "ligotimegps",
        "glue.lal",
    ),
)))

if TYPE_CHECKING:
    from glue.lal import LIGOTimeGPS as _Glue_LIGOTimeGPS
    from lal import LIGOTimeGPS as _LAL_LIGOTimeGPS
    from ligotimegps import LIGOTimeGPS as _Ligotimegps_LIGOTimeGPS

    GpsType = (
        _Ligotimegps_LIGOTimeGPS
        | _Glue_LIGOTimeGPS
        | _LAL_LIGOTimeGPS
    )
