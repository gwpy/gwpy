# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""This module attaches the HDF5 input output methods to the TimeSeries.
"""

from ...io import registry as io_registry
from ...io.hdf5 import (identify_hdf5, with_read_hdf5, with_write_hdf5)
from ...types.io.hdf5 import (read_hdf5_array, write_hdf5_array)
from .. import (TimeSeries, TimeSeriesDict,
                StateVector, StateVectorDict)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- read ---------------------------------------------------------------------

def read_hdf5_timeseries(f, path=None, start=None, end=None, **kwargs):
    # read data
    kwargs.setdefault('array_type', TimeSeries)
    ts = read_hdf5_array(f, path=path, **kwargs)
    # crop if needed
    if start is not None or end is not None:
        return ts.crop(start, end)
    else:
        return ts


@with_read_hdf5
def read_hdf5_dict(h5f, names=None, group=None, **kwargs):
    """Read a `TimeSeriesDict` from HDF5
    """
    # find group from which to read
    if group:
        h5g = h5f[group]
    else:
        h5g = h5f

    # find list of names to read
    if names is None:
        # TODO: improve the TimeSeries -> HDF5 format to make detecting
        #       a TimeSeries easier
        names = [key for key in h5g if 'dx' in h5g[key]]

    # read names
    out = kwargs.pop('dict_type', TimeSeriesDict)()
    kwargs.setdefault('array_type', out.EntryClass)
    for name in names:
        out[name] = read_hdf5_timeseries(h5g[name], **kwargs)

    return out


def read_hdf5_factory(data_class):
    if issubclass(data_class, dict):
        def read_(*args, **kwargs):
            kwargs.setdefault('dict_type', data_class)
            return read_hdf5_dict(*args, **kwargs)
    else:
        def read_(*args, **kwargs):
            kwargs.setdefault('array_type', data_class)
            return read_hdf5_timeseries(*args, **kwargs)

    return read_


# -- write --------------------------------------------------------------------

@with_write_hdf5
def write_hdf5_dict(tsdict, h5f, group=None, **kwargs):
    """Write a `TimeSeriesBaseDict` to HDF5

    Each series in the dict is written as a dataset in the group
    """
    # create group if needed
    if group and group not in h5f:
        h5g = h5f.create_group(group)
    elif group:
        h5g = h5f[group]
    else:
        h5g = h5f

    # write each timeseries
    for key, series in tsdict.items():
        series.write(h5g, path=str(key), **kwargs)


# -- register -----------------------------------------------------------------

# series classes
for series_class in (TimeSeries, StateVector):
    reader = read_hdf5_factory(series_class)
    io_registry.register_reader('hdf5', series_class, reader)
    io_registry.register_writer('hdf5', series_class, write_hdf5_array)
    io_registry.register_identifier('hdf5', series_class, identify_hdf5)

# dict classes
for dict_class in (TimeSeriesDict, StateVectorDict):
    reader = read_hdf5_factory(dict_class)
    io_registry.register_reader('hdf5', dict_class, reader)
    io_registry.register_writer('hdf5', dict_class, write_hdf5_dict)
    io_registry.register_identifier('hdf5', dict_class, identify_hdf5)
