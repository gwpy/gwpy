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

"""Read a `Series` from AN ASCII file

These files should be in two-column x,y format
"""

from numpy import (savetxt, loadtxt)

from astropy.io.registry import (register_reader,
                                 register_writer,
                                 register_identifier)

from ..data import Series
from .. import version
from .utils import identify_factory

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

def read_ascii(filepath, _obj=Series, xcol=0, ycol=1, delimiter=None, **kwargs):
    """Read a `Series` from an ASCII file
    """
    # get specific args for loadtxt
    loadargs = {'unpack': True, 'usecols': [xcol, ycol]}
    for kwarg in ['dtype', 'comments', 'delimiter', 'converters', 'skiprows']:
        if kwarg in kwargs:
            loadargs[kwarg] = kwargs.pop(kwarg)
    # read data, format and return
    x, y = loadtxt(filepath, **loadargs)
    return _obj(y, index=x, **kwargs)


def write_ascii(series, fobj, fmt='%.18e', delimiter=' ', newline='\n',
                header='', footer='', comments='# '):
    """Write a `Series` to a file in ASCII format

    Parameters
    ----------
    series : :class:`~gwpy.data.Series`
        data series to write
    fobj : `str`, `file`
        file object, or path to file, to write to

    See also
    --------
    numpy.savetxt : for documentation of keyword arguments
    """
    x = series.index.data
    y = series.data
    return savetxt(fobj, zip(x, y), fmt=fmt, delimiter=delimiter,
                   newline=newline, header=header, footer=footer,
                   comments=comments)


formats = {'txt': None,
           'csv': ','}

def ascii_io_factory(obj, delimiter=None):
    def _read(filepath, **kwargs):
        kwargs.setdefault('delimiter', delimiter)
        return read_ascii(filepath, _obj=obj,**kwargs)
    def _write(series, filepath, **kwargs):
        kwargs.setdefault('delimiter', delimiter or ' ')
        return write_ascii(series, filepath, **kwargs)
    return _read, _write


def register_ascii(obj):
    """Register ASCII I/O methods for given type obj

    This factory method registers 'txt' and 'csv' I/O formats with
    a reader, writer, and auto-identifier
    """
    for form, delim in formats.iteritems():
        read_, write_ = ascii_io_factory(obj, delim)
        register_identifier(form, obj, identify_factory(form))
        register_writer(form, obj, write_)
        register_reader(form, obj, read_)
