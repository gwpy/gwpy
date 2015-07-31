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

While these methods are avialable as methods of the class itself,
this module attaches them to the unified I/O registry, making it a bit
cleaner.
"""

from astropy.io.registry import (register_reader, register_writer,
                                 register_identifier)

from ... import version
from ...io.hdf5 import identify_hdf5
from .. import (TimeSeries, StateVector)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

register_reader('hdf', TimeSeries, TimeSeries.from_hdf5)
register_writer('hdf', TimeSeries, TimeSeries.to_hdf5)
register_identifier('hdf', TimeSeries, identify_hdf5)

register_reader('hdf', StateVector, StateVector.from_hdf5)
register_writer('hdf', StateVector, StateVector.to_hdf5)
register_identifier('hdf', StateVector, identify_hdf5)
