# -*- coding: utf-8 -*-
# Copyright (C) Patrick Godwin (2019)
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

import re
import os

import numpy

import h5py

from astropy.io.misc.hdf5 import read_table_hdf5
from astropy.table import vstack

from ...io.hdf5 import with_read_hdf5
from ...io.registry import register_reader
from .. import EventTable
from ..filter import filter_table

__author__ = 'Patrick Godwin <patrick.godwin@ligo.org>'


@with_read_hdf5
def table_from_file(source, channel, columns=None, selection=None):
    """Read an `EventTable` from a SNAX HDF5 file

    Parameters
    ----------
    source : `h5py.File`
        the file path of open `h5py` object from which to read the data

    channel : `str`
        the channel to read data from

    columns : `list` of `str`, optional
        the list of column names to read

    selection : `str`, or `list` of `str`, optional
        one or more column filters with which to downselect the
        returned table rows as they as read, e.g. ``'snr > 5'``;
        multiple selections should be connected by ' && ', or given as
        a `list`, e.g. ``'snr > 5 && frequency < 1000'`` or
        ``['snr > 5', 'frequency < 1000']``

    Returns
    -------
    table : `~gwpy.table.EventTable`
    """

    # concatenate all datasets with a given channel
    datasets = source[channel].keys()
    tables = [read_table_hdf5(source, path=os.path.join(channel, dataset))
              for dataset in datasets]
    table = vstack(tables)

    # apply selection and column filters
    if selection:
        table = filter_table(table, selection)
    if columns:
        table = table[columns]

    return EventTable(data=table)


# register for unified I/O
register_reader('hdf5.snax', EventTable, table_from_file)
