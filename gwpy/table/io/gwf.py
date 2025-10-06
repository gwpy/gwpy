# Copyright (c) 2017 Louisiana State University
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

"""Read events from GWF FrEvent structures into a Table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy.table import Table

from ...io import gwf as io_gwf
from ...io.utils import FileLike
from ...time import LIGOTimeGPS
from .. import EventTable
from ..filter import parse_column_filters

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import IO

    from LDASTools.frameCPP import FrEvent

    from ..filter import FilterSpec

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- read ----------------------------

def _columns_from_frevent(frevent: FrEvent) -> list[str]:
    """Get list of column names from frevent."""
    params = dict(frevent.GetParam())
    return [
        "time",
        "amplitude",
        "probability",
        "timeBefore",
        "timeAfter",
        "comment",
        *list(params.keys()),
    ]


def _row_from_frevent(
    frevent: FrEvent,
    columns: Iterable[str],
    where: Iterable[FilterSpec],
) -> list[str | float] | None:
    """Generate a table row from an FrEvent.

    Filtering (``where``) is done here, rather than in the table reader,
    to enable filtering on columns that aren't being returned.

    Returns `None` if this event doesn't match the ``where`` conditions.
    """
    # read params
    params = dict(frevent.GetParam())
    params["time"] = float(LIGOTimeGPS(*frevent.GetGTime()))
    params["amplitude"] = frevent.GetAmplitude()
    params["probability"] = frevent.GetProbability()
    params["timeBefore"] = frevent.GetTimeBefore()
    params["timeAfter"] = frevent.GetTimeAfter()
    params["comment"] = frevent.GetComment()

    # filter
    if not all(op_(params[c], t) for c, op_, t in where):
        return None

    # return event as list
    return [params[c] for c in columns]


def table_from_gwf(
    filename: str | Path | IO,
    name: str,
    columns: Iterable[str] | None = None,
    where: str | list[str] | None = None,
) -> Table:
    """Read a Table from FrEvent structures in a GWF file (or files).

    This method requires |LDAStools.frameCPP|_.

    Parameters
    ----------
    filename : `str`, `pathlib.Path`, `file`
        The path of GWF file to read.

    name : `str`
        The name associated with the `FrEvent` structures.

    columns : `list` of `str`
        List of column names to read.

    where : `str`, `list` of `str`
        One or more filter condition strings to apply, e.g. ``'snr>6'``.

    Returns
    -------
    Table
        A table of rows read from the GWF input.
    """
    gwf_framecpp = io_gwf.import_backend("frameCPP")

    # open frame file
    if isinstance(filename, FileLike):
        filename = filename.name
    stream = gwf_framecpp.open_gwf(filename)

    # parse where conditions and map to column indices
    if where is None:
        where = []
    conditions = parse_column_filters(where)

    # read events row by row
    data = []
    i = 0
    while True:
        try:
            frevent = stream.ReadFrEvent(i, name)
        except IndexError:
            break
        i += 1
        # read first event to get column names
        if columns is None:
            columns = _columns_from_frevent(frevent)
        # read row with filter
        row = _row_from_frevent(frevent, columns, conditions)
        if row is not None:  # if passed condition
            data.append(row)

    return Table(rows=data, names=columns)


# -- write ---------------------------

def table_to_gwf(
    table: Table,
    filename: str | Path,
    name: str,
    **kwargs,
) -> None:
    """Create a new `~frameCPP.FrameH` and fill it with data.

    Parameters
    ----------
    table : `~astropy.table.Table`
        the data to write

    filename : `str`
        the name of the file to write into

    name: `str`, optional
        The name to give the ``FrameH`` object **and** each ``FrEvent``.

    kwargs
        Other keyword arguments are passed to the GWF creator and writer
        functions (see below).

    See Also
    --------
    gwpy.io.gwf.create_frame
        For details of how the GWF ``FrameH`` structure is created and
        any valid keyword arguments.

    gwpy.io.gwf.write_frames
        For details of the GWF ``FrEvent`` structures are created, and
        any valid keyword arguments.
    """
    from LDAStools.frameCPP import (
        FrEvent,
        GPSTime,
    )

    gwf_framecpp = io_gwf.import_backend("frameCPP")

    # create frame
    write_kw = {
        key: kwargs.pop(key)
        for key in (
            "compression",
            "compression_level",
        ) if key in kwargs
    }
    frame = gwf_framecpp.create_frame(
        name=name,
        **kwargs,
    )

    # append row by row
    names = table.dtype.names
    for row in table:
        rowd = {n: row[n] for n in names}
        gps = LIGOTimeGPS(rowd.pop("time", 0))
        frame.AppendFrEvent(FrEvent(
            str(name),
            str(rowd.pop("comment", "")),
            str(rowd.pop("inputs", "")),
            GPSTime(gps.gpsSeconds, gps.gpsNanoSeconds),
            float(rowd.pop("timeBefore", 0)),
            float(rowd.pop("timeAfter", 0)),
            int(rowd.pop("eventStatus", 0)),
            float(rowd.pop("amplitude", 0)),
            float(rowd.pop("probability", -1)),
            str(rowd.pop("statistics", "")),
            list(rowd.items()),  # remaining params as tuple
        ))

    # write frame to file
    gwf_framecpp.write_frames(
        filename,
        [frame],
        **write_kw,
    )


# -- registration --------------------

for klass in (Table, EventTable):
    # read
    klass.read.registry.register_reader(
        "gwf",
        klass,
        table_from_gwf,
    )
    # write
    klass.write.registry.register_writer(
        "gwf",
        klass,
        table_to_gwf,
    )
    # identify
    klass.read.registry.register_identifier(
        "gwf",
        klass,
        io_gwf.identify_gwf,
    )
