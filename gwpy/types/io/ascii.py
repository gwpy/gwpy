# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""Read a `Series` from AN ASCII file

These files should be in two-column x,y format
"""

from numpy import (savetxt, loadtxt, column_stack)

from ...io import registry as io_registry
from ...io.utils import identify_factory
from .. import Series

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- read ---------------------------------------------------------------------

def read_ascii_series(input_, array_type=Series, unpack=True, **kwargs):
    """Read a `Series` from an ASCII file

    Parameters
    ----------
    input : `str`, `file`
        file to read

    array_type : `type`
        desired return type
    """
    xarr, yarr = loadtxt(input_, unpack=unpack, **kwargs)
    return array_type(yarr, xindex=xarr)


# -- write --------------------------------------------------------------------

def write_ascii_series(series, output, **kwargs):
    """Write a `Series` to a file in ASCII format

    Parameters
    ----------
    series : :class:`~gwpy.data.Series`
        data series to write

    output : `str`, `file`
        file to write to

    See also
    --------
    numpy.savetxt
        for documentation of keyword arguments
    """
    xarr = series.xindex.value
    yarr = series.value
    return savetxt(output, column_stack((xarr, yarr)), **kwargs)


# -- register -----------------------------------------------------------------

def register_ascii_series_io(array_type, format='txt', identify=True,
                             **defaults):
    """Register ASCII read/write/identify methods for the given array
    """
    def _read(filepath, **kwargs):
        kwgs = defaults.copy()
        kwgs.update(kwargs)
        return read_ascii_series(filepath, array_type=array_type, **kwgs)

    def _write(series, output, **kwargs):
        kwgs = defaults.copy()
        kwgs.update(kwargs)
        return write_ascii_series(series, output, **kwgs)

    io_registry.register_reader(format, array_type, _read)
    io_registry.register_writer(format, array_type, _write)
    if identify:
        io_registry.register_identifier(format, array_type,
                                        identify_factory(format))
