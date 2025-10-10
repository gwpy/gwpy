# Copyright (c) 2019-2020 Patrick Godwin
#               2020 Cardiff University
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

"""Read events from SNAX."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.table import vstack

from ...io.hdf5 import with_read_hdf5
from .. import EventTable
from .utils import (
    read_with_columns,
    read_with_where,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal

    import h5py
    from astropy.table import Table

__author__ = "Patrick Godwin <patrick.godwin@ligo.org>"


@read_with_columns
@read_with_where
@with_read_hdf5
def table_from_file(
    source: h5py.Group,
    channels: Iterable[str] | None = None,
    on_missing: Literal["error", "warn", "ignore"] = "error",
    *,
    compact: bool = False,
) -> Table:
    """Read an `EventTable` from a SNAX HDF5 file.

    Parameters
    ----------
    source : `str`, `pathlib.Path`, `file`, `h5py.File`
        The file path of open `h5py` object from which to read the data.

    channels : `str` or `list`, optional
        The channel(s) to read data from. if no channels selected,
        read in all channels.

    on_missing: `str`, optional
        How to proceed when channels requested are missing. One of:

        "warn"
            Emit a warning when missing channels are discovered.

        "error"
            Raise an exception.

        Default is ``"error"``

    compact : `bool`, optional
        Whether to store a compact integer representation in the channel
        column rather than the full channel name, instead storing a mapping
        (`channel_map`) in the table metadata.
        Default is `False`.

    Returns
    -------
    table : `~astropy.table.Table`
    """
    # format channels appropriately
    if isinstance(channels, str):
        channels = [channels]

    # only query channels contained in file
    # if channels not specified, load all of them
    if channels is None:
        channels = source.keys()
    else:
        channels = set(channels)
        found = set(source.keys())
        missing = channels - found
        # check whether missing channels should
        # be an error or simply a warning
        if missing and on_missing != "ignore":
            msg = "requested channels not found in SNAX file: '{}'".format(
                "', '".join(missing),
            )
            if on_missing == "error":
                raise ValueError(msg)
            if on_missing == "warn":
                warnings.warn(msg, stacklevel=2)
            else:
                msg = "on_missing argument must be one of 'error', 'warn', or 'ignore'"
                raise ValueError(msg)
        channels = channels & found

    # read data, adding in 'channel' column
    # to preserve uniqueness across channels
    tables = []
    for channel in channels:
        table = vstack(
            list(map(read_table_hdf5, source[channel].values())),
            join_type="exact",
            metadata_conflicts="error",
        )
        # determine whether to store a compact
        # representation of the channel, storing
        # the mapping to table metadata instead
        if compact:
            table["channel"] = hash(channel)
            table.meta["channel_map"] = {hash(channel): channel}
        else:
            table["channel"] = channel
        tables.append(table)

    # combine results
    return vstack(tables, join_type="exact", metadata_conflicts="error")


# register for unified I/O
EventTable.read.registry.register_reader(
    "hdf5.snax",
    EventTable,
    table_from_file,
)
