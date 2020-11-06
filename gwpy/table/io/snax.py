# -*- coding: utf-8 -*-
# Copyright (C) Patrick Godwin (2019-2020)
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

"""Read events from SNAX
"""

from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.table import vstack

from ...io.hdf5 import with_read_hdf5
from ...io.registry import register_reader
from .. import EventTable
from .utils import (read_with_columns, read_with_selection)

__author__ = 'Patrick Godwin <patrick.godwin@ligo.org>'


@read_with_columns
@read_with_selection
@with_read_hdf5
def table_from_file(source, channels=None):
    """Read an `EventTable` from a SNAX HDF5 file

    Parameters
    ----------
    source : `h5py.File`
        the file path of open `h5py` object from which to read the data

    channels : `str` or `list`
        the channel(s) to read data from

    Returns
    -------
    table : `~gwpy.table.EventTable`
    """
    # format channels appropriately
    if isinstance(channels, str):
        channels = [channels]

    # only query channels contained in file
    # if channels not specified, load all of them
    if channels is None:
        channels = source.keys()
    else:
        channels = set(channels) & set(source.keys())

    # read data, adding in 'channel' column
    # to preserve uniqueness across channels
    tables = []
    for channel in channels:
        table = vstack(
            list(map(read_table_hdf5, source[channel].values())),
            join_type="exact",
            metadata_conflicts="error",
        )
        table["channel"] = channel
        tables.append(table)

    # combine results
    return vstack(tables, join_type="exact", metadata_conflicts="error")


# register for unified I/O
register_reader('hdf5.snax', EventTable, table_from_file)
