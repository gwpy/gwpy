# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013-2016)
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

"""Read LIGO_LW documents into glue.ligolw.table.Table objects.
"""

import warnings

from glue.ligolw.table import StripTableName as strip
from glue.ligolw.lsctables import TableByName

from ...io import registry
from ...io.cache import (read_cache, file_list)
from ...io.ligolw import (table_from_file, identify_ligolw)
from .. import GWRecArray

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = []


def read_table_factory(table_):
    """Define a custom function to read this table from a LIGO_LW file.
    """
    def _read_ligolw(f, *args, **kwargs):
        return table_from_file(f, table_.tableName, *args, **kwargs)

    def _read_recarray(f, *args, **kwargs):
        # set up keyword arguments
        reckwargs = {
            'on_attributeerror': 'raise',
            'get_as_columns': False
        }
        for key in reckwargs:
            if key in kwargs:
                reckwargs[key] = kwargs.pop(key)
        reckwargs['columns'] = kwargs.get('columns', None)

        # handle multiprocessing
        nproc = kwargs.pop('nproc', 1)
        if nproc > 1:
            kwargs['format'] = strip(table_.tableName)
            return read_cache(file_list(f), GWRecArray, nproc, None,
                              *args, **kwargs)

        return table_from_file(f, table_.tableName,
                               *args, **kwargs).to_recarray(**reckwargs)

    return _read_ligolw, _read_recarray


# register reader and auto-id for LIGO_LW
for table in TableByName.itervalues():
    tablename = strip(table.tableName)
    llwfunc, recfunc = read_table_factory(table)
    # register generic reader and table-specific reader for LIGO_LW
    registry.register_reader('ligolw', table, llwfunc)
    registry.register_reader(tablename, table, llwfunc)
    registry.register_identifier('ligolw', table, identify_ligolw)
    # register table-specific reader for GWRecArray
    registry.register_reader(tablename, GWRecArray, recfunc)
