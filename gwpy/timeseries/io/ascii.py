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

"""ASCII I/O registrations for gwpy.timeseries objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...types.io.ascii import (
    read_ascii_series,
    register_ascii_io,
)
from .. import (
    StateVector,
    TimeSeries,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

    from ...time import SupportsToGps


def read_ascii(
    input_: str | Path | IO,
    array_type: type = TimeSeries,
    *,
    unpack: bool = True,
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    **kwargs,
) -> TimeSeries:
    """Read a `TimeSeries` from an ASCII file.

    Parameters
    ----------
    input : `str`, `file`
        File to read.

    array_type : `type`
        Desired return type.

    start : `float`, `astropy.units.Quantity`, optional
        The desired start point of the X-axis, defaults to
        the start point of the incoming series.

    end : `float`, `astropy.units.Quantity`, optional
        The desired end point of the X-axis, defaults to
        the end point of the incoming series.

    kwargs
        All other keyword arguments are passed to
        `numpy.loadtxt`.

    Returns
    -------
    series : instance of ``array_type``
        The data series as read from the input file.

    See Also
    --------
    gwpy.types.io.ascii.read_ascii_series
        For details of how the series is read from the file.
    """
    ts = read_ascii_series(
        input_,
        array_type=array_type,
        unpack=unpack,
        **kwargs,
    )
    return ts.crop(start=start, end=end)


for series_class in (
    TimeSeries,
    StateVector,
):
    register_ascii_io(
        series_class,
        format="txt",
        reader=read_ascii,
    )
    register_ascii_io(
        series_class,
        format="csv",
        reader=read_ascii,
        delimiter=",",
    )
