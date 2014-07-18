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

"""Utilities for LIGO_LW tables.
"""

import re

import numpy

from . import lsctables
from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

EVENT_TABLES = (lsctables.SnglBurstTable,
                lsctables.MultiBurstTable,
                lsctables.SnglInspiralTable,
                lsctables.MultiInspiralTable,
                lsctables.SnglRingdownTable)


def get_table_column(table, column, dtype=numpy.dtype(float)):
    """Extract a column from the given table.

    This method uses the following logic to determine how to extract a
    column:

    - if the table has a 'get_column' method, use that,
    - if column = 'time', use one of the ``get_start``, ``get_peak`` or
      ``get_end`` methods, depending on the table type, or
    - get the column using the name as given

    Parameters
    ----------
    table : :class:`glue.ligolw.table.Table`
        the LIGO_LW Table from which to extract a column.
    column : `str`
        the name of the column to find.
    dtype : :class:`~numpy.dtype`, `type`, optional
        the requested data type of the returned array.

    Returns
    -------
    array : :class:`numpy.ndarray`
        an array containing the data from the requested column
    """
    column = str(column).lower()
    if hasattr(table, 'get_%s' % column):
        return numpy.asarray(getattr(table, 'get_%s' % column)()).astype(dtype)
    elif column == 'time':
        if re.match('(sngl_inspiral|multi_inspiral)', table.tableName, re.I):
            return numpy.asarray(table.get_end()).astype(dtype)
        elif re.match('(sngl_burst|multi_burst)', table.tableName, re.I):
            return numpy.asarray(table.get_peak()).astype(dtype)
        elif re.match('(sngl_ring|multi_ring)', table.tableName, re.I):
            return numpy.asarray(table.get_start()).astype(dtype)
        elif re.match('sim_burst', table.tableName, re.I):
            return numpy.asarray(get_table_column(table, 'time_geocent_gps') +
                                 get_table_column(table, 'time_geocent_gps_ns')
                                 * 1e-9)
    if hasattr(table, 'get_column'):
        return numpy.asarray(table.get_column(column)).astype(dtype)
    else:
        return numpy.asarray(table.getColumnByName(column)).astype(dtype)


def get_row_value(row, attr):
    """Get the attribute value of a given LIGO_LW row.

    Parameters
    ----------
    row : `object`
        a row of a LIGO_LW `Table`.
    attr : `str`
        the name of the column attribute to retrieve.

    See Also
    --------
    get_table_column : for details on the column-name logic
    """
    attr = str(attr).lower()
    cname = row.__class__.__name__
    if hasattr(row, 'get_%s' % attr):
        return getattr(row, 'get_%s' % attr)()
    elif attr == 'time':
        if re.match('(Sngl|Multi)Inspiral', cname, re.I):
            return row.get_end()
        elif re.match('(Sngl|Multi)Burst', cname, re.I):
            return row.get_peak()
        elif re.match('(Sngl|Multi)Ring', cname, re.I):
            return row.get_start()
    else:
        return getattr(row, attr)
