# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""This module defines the `GWTable`.
"""

from itertools import imap

from numpy import (dtype, fromiter)

from astropy.table import (Column, Table)

from glue.ligolw import types as ligolwtypes
from glue.ligolw.table import Table as LigoLwTable
from glue.ligolw.ilwd import ilwdchar as ILWDChar

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version
__all__ = ['GWTable']


class GWTable(Table):
    """A :mod:`data table <astropy.table>` holding GW data.
    """
    @classmethod
    def from_ligolw(cls, llwtable):
        """Define a new GWTable to mirror a LIGO_LW table.
        """
        # define new table
        colnames = map(str, llwtable.columnnames)
        dtypes = map(_numpy_dtype, llwtable.columntypes)
        new = cls()#names=colnames, dtype=dtypes)
        new.meta['name'] = llwtable.tableName
        # seed table if given an instance, rather than a class
        for name, dtype_ in zip(colnames, dtypes):
            int(name, dtype_)
            try:
                coldata = fromiter(llwtable.getColumnByName(name), dtype=dtype_)
            except TypeError:
                coldata = fromiter(imap(dtype_.type, llwtable.getColumnByName(name)), dtype=dtype_)
            print(coldata)
            new.add_column(Column(coldata, name=name, dtype=dtype_))
        return new


def _numpy_dtype(typestr):
    """Convert a LIGO_LW XML data type string into numpy format.
    """
    # get LIGO_LW mapping
    try:
        return dtype(ligolwtypes.ToNumPyType[typestr])
    except KeyError:
        pass
    if typestr in ligolwtypes.IDTypes:
        return dtype(int)
    elif typestr in ligolwtypes.StringTypes:
        return dtype('<U2')
    try:
        return dtype(ligolwtypes.ToPyType[typestr])
    except (KeyError, TypeError):
        pass
    raise ValueError("Cannot convert type '%s' to numpy.dtype" % typestr)


