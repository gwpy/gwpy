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
from ...io.hdf5 import identify_hdf5
from ...types.io.hdf5 import (read_hdf5_array, write_hdf5_array)
from .. import (TimeSeries, StateVector)

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


def read_hdf5_statevector(*args, **kwargs):
    kwargs['array_type'] = StateVector
    return read_hdf5_timeseries(*args, **kwargs)


# -- register -----------------------------------------------------------------

# TimeSeries
io_registry.register_reader('hdf5', TimeSeries, read_hdf5_timeseries)
io_registry.register_writer('hdf5', TimeSeries, write_hdf5_array)
io_registry.register_identifier('hdf5', TimeSeries, identify_hdf5)

# StateVector
io_registry.register_reader('hdf5', StateVector, read_hdf5_statevector)
io_registry.register_writer('hdf5', StateVector, write_hdf5_array)
io_registry.register_identifier('hdf5', StateVector, identify_hdf5)
