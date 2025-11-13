# Copyright (c) 2020-2025 Cardiff University
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

"""Read gravitational-wave frame (GWF) files using the FrameL API.

The frame format is defined in :dcc:`LIGO-T970130`.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    TypedDict,
)

import framel
from igwn_segments import infinity

from ....io.gwf.core import _series_name
from ....io.utils import (
    file_list,
    file_path,
)
from ....segments import Segment
from ... import (
    TimeSeries,
    TimeSeriesBaseList,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import (
        IO,
        TypeVar,
    )

    import numpy

    from ....time import LIGOTimeGPS
    from ... import (
        TimeSeriesBase,
        TimeSeriesBaseDict,
    )

    _TimeSeriesType = TypeVar("_TimeSeriesType", bound=TimeSeriesBase)

inf = infinity()

FRAMEL_COMPRESSION_GZIP = {
    "gzip",
    1,
    257,
    None,
}

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- read ----------------------------

def read(
    source: str | Path | IO,
    channels: list[str],
    start: LIGOTimeGPS | None = None,
    end: LIGOTimeGPS | None = None,
    scaled: bool | None = None,
    type: str | dict[str, str] | None = None,
    series_class: type[TimeSeriesBase] = TimeSeries,
) -> TimeSeriesBaseDict:
    """Read data from one or more GWF files using the FrameL API."""
    # scaled must be provided to provide a consistent API with frameCPP
    if scaled is not None:
        warnings.warn(
            "the `scaled` keyword argument is not supported by framel, "
            "if you require ADC scaling, please install "
            "python-ldas-tools-framecpp",
            stacklevel=2,
        )

    # parse input source
    files = file_list(source)

    # read each file and channel individually and append
    tmp: dict[str, TimeSeriesBaseList] = defaultdict(TimeSeriesBaseList)
    for file_ in files:
        for series in _read_gwf(
            file_,
            channels,
            start,
            end,
            series_class,
        ):
            tmp[series.name].append(series)
    out = series_class.DictClass()
    for channel, serieslist in tmp.items():
        out[channel] = serieslist.join()

    return out


def _read_gwf(
    filename: str,
    channels: list[str],
    start: LIGOTimeGPS | None,
    end: LIGOTimeGPS | None,
    series_class: type[TimeSeriesBase],
) -> Iterator[TimeSeriesBase]:
    """Read data from a single file."""
    if start is None:
        start = -inf
    if end is None:
        end = inf
    span = Segment(start, end)
    record: set[str] = set()
    _record = record.add
    for name in channels:
        # framel.frgetvect will silently pad available data to
        # whatever span you gave it without error, so we have to
        # read everything and crop
        new = _read_channel(
            filename,
            name,
            -1,  # read from start
            -1,  # read all data
            series_class,
        )
        try:
            overlap = new.span & span
        except ValueError:
            # this channel doesn't overlap with the requested span
            continue
        if (keep := new.crop(*overlap)).size:
            yield keep
            _record(keep.name)

    # if any channels weren't read, something went wrong
    for channel in channels:
        if (name := str(channel)) not in record:
            msg = f"cannot read data for '{name}' from '{filename}' in interval {span}"
            raise ValueError(msg)


def _read_channel(
    filename: str | Path,
    channel: str,
    start: float,
    duration: float,
    series_class: type[_TimeSeriesType],
) -> _TimeSeriesType:
    """Read one channel from one file."""
    try:
        data, gps, offset, dx, xunit, yunit = framel.frgetvect1d(
            str(filename),
            str(channel),
            start=start,
            span=duration,
        )
    except KeyError as exc:  # upstream errors
        raise ValueError(str(exc)) from exc
    except ValueError as exc:
        if str(exc) == "NULL pointer access":
            msg = f"channel '{channel}' not found"
            raise ValueError(msg) from exc
        raise
    return series_class(
        data,
        name=channel,
        x0=gps+offset,
        dx=dx,
        xunit=xunit,
        unit=yunit,
    )


# -- write ---------------------------

def write(
    tsdict: TimeSeriesBaseDict,
    outfile: str | Path | IO,
    start: LIGOTimeGPS,
    end: LIGOTimeGPS,
    type: str | None = None,
    name: str | None = None,
    run: int = 0,
    compression: int | str | None = None,
    compression_level: int | None = None,
) -> None:
    """Write data to a GWF file using the FrameL API."""
    if name is not None:
        warnings.warn(
            "python-framel does not support setting FrHistory 'name', "
            "this value will be ignored",
            stacklevel=2,
        )
    if run:
        warnings.warn(
            "python-framel does not support setting FrHistory 'run', "
            "this value will be ignored",
            stacklevel=2,
        )

    # FrameL Python bindings only support Gzip
    if isinstance(compression, str):
        compression = compression.lower()
    if compression not in FRAMEL_COMPRESSION_GZIP:
        msg = "python-framel only supports compression='GZIP'"
        raise ValueError(msg)
    if compression_level not in (None, 1):
        warnings.warn(
            "python-framel only supports GZIP level 1",
            stacklevel=1,
        )

    # format and crop each series
    channellist: list[FrameLVectDict] = [
        _channel_data_to_write(
            series.crop(start=start, end=end),
            type,
        ) for series in tsdict.values()
    ]
    return framel.frputvect(file_path(outfile), channellist)


class FrameLVectDict(TypedDict):
    """Dict of channel vector information."""

    name: str
    data: numpy.ndarray
    start: float
    dx: float
    x_unit: str
    y_unit: str
    kind: str
    type: int
    subType: int


def _channel_data_to_write(
    timeseries: TimeSeriesBase,
    type_: str | None,
) -> FrameLVectDict:
    """Format a series into a dict suitable for `framel.frputvect`."""
    return FrameLVectDict({
        "name": _series_name(timeseries) or "",
        "data": timeseries.value,
        "start": timeseries.x0.to("s").value,
        "dx": timeseries.dx.to("s").value,
        "x_unit": str(timeseries.xunit),
        "y_unit": str(timeseries.unit),
        "kind": (
            type_
            or getattr(timeseries.channel, "_ctype", "proc")
            or "proc"
        ).upper(),
        "type": 1,
        "subType": 0,
    })
