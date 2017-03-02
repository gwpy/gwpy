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

.. warning::

   This module has been totally deprecated in favour of the
    `EventTable` object and the methods provided with it.

   This module will be removed before the 1.0 release.
"""

import warnings
warnings.warn('The gwpy.table.utils module has been deprecated and will be '
              'removed prior to the 1.0 release.', DeprecationWarning)

# -----------------------------------------------------------------------------
#
# -- DEPRECATED - remove before 1.0 release -----------------------------------
#
# -----------------------------------------------------------------------------

import re

import numpy

from . import lsctables

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

EVENT_TABLES = (lsctables.SnglBurstTable,
                lsctables.MultiBurstTable,
                lsctables.SnglInspiralTable,
                lsctables.MultiInspiralTable,
                lsctables.SnglRingdownTable)

TIME_COLUMN = {
    lsctables.CoincInspiralTable.tableName: ('end_time', 'end_time_ns'),
    lsctables.CoincRingdownTable.tableName: ('start_time', 'start_time_ns'),
    lsctables.MultiBurstTable.tableName: ('peak_time', 'peak_time_ns'),
    lsctables.MultiInspiralTable.tableName: ('end_time', 'end_time_ns'),
    lsctables.SimBurstTable.tableName: ('time_geocent_gps',
                                        'time_geocent_gps_ns'),
    lsctables.SimInspiralTable.tableName: ('geocent_end_time',
                                           'geocent_end_time_ns'),
    lsctables.SimRingdownTable.tableName: ('geocent_start_time',
                                           'geocent_start_time_ns'),
    lsctables.SnglBurstTable.tableName: ('peak_time', 'peak_time_ns'),
    lsctables.SnglInspiralTable.tableName: ('end_time', 'end_time_ns'),
    lsctables.SnglRingdownTable.tableName: ('start_time', 'start_time_ns'),
}


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
    # if it has a special get_ method for this column, use it
    if hasattr(table, 'get_%s' % column):
        return numpy.asarray(getattr(table, 'get_%s' % column)()).astype(dtype)
    # otherwise if asked for 'time' and is a recarray
    if column == 'time' and isinstance(table, numpy.recarray):
        try:
            return get_rec_time(table)
        except ValueError:
            pass
    # otherwise if asked for 'time' and is a LIGO_LW table
    if column == 'time' and type(table) in EVENT_TABLES:
        if re.match('(sngl_inspiral|multi_inspiral)', table.tableName, re.I):
            return numpy.asarray(table.get_end()).astype(dtype)
        if re.match('(sngl_burst|multi_burst)', table.tableName, re.I):
            return numpy.asarray(table.get_peak()).astype(dtype)
        if re.match('(sngl_ring|multi_ring)', table.tableName, re.I):
            return numpy.asarray(table.get_start()).astype(dtype)
        if re.match('sim_burst', table.tableName, re.I):
            return numpy.asarray(get_table_column(table, 'time_geocent_gps') +
                                 get_table_column(table, 'time_geocent_gps_ns')
                                 * 1e-9)
    # try and use get_column
    if hasattr(table, 'get_column'):
        return numpy.asarray(table.get_column(column)).astype(dtype)
    # otherwise use the LIGO_LW DOM API column getter
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
    # shortcut from recarray
    if isinstance(row, numpy.void) and attr == 'time':
        return get_rec_time(row)
    if isinstance(row, numpy.void):
        return row[attr]
    # presume ligolw row instance
    attr = str(attr).lower()
    cname = type(row).__name__
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


def get_rec_time(table):
    """Get the 'time' from a `numpy.recarray` or `numpy.record`
    """
    # if 'time' not present, try and get LIGO_LW-style INT+NS value
    if 'time' not in table.dtype.fields:
        if 'end_time' and 'end_time_ns' in table.dtype.fields:
            return table['end_time'] + table['end_time_ns'] * 1e-9
        if 'peak_time' and 'peak_time_ns' in table.dtype.fields:
            return table['peak_time'] + table['peak_time_ns'] * 1e-9
        if 'start_time' and 'start_time_ns' in table.dtype.fields:
            return table['start_time'] + table['start_time_ns'] * 1e-9
    # if that didn't work, just return time
    return table['time']
