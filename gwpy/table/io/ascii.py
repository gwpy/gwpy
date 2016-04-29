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

"""Read event tables from ASCII files.

This module only defines a function factory for reading lines of ASCII
into a particular LIGO_LW table object.

Each specific ASCII table format should define their own line parser
(that generates each row of the table) and pass it to the factory method.
"""

from six import string_types

from numpy import loadtxt

from glue.ligolw.table import (reassign_ids, StripTableName)

from ..lsctables import (New, TableByName)
from ..utils import TIME_COLUMN
from ...io.cache import file_list
from ...io.utils import identify_factory
from ...io.registry import (register_reader, register_identifier)
from ...time import LIGOTimeGPS

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def return_reassign_ids(elem):
    """Wrapper to `glue.ligolw.table.reassign_ids` that returns
    """
    reassign_ids(elem)
    return elem


def table_from_ascii_factory(table, format, trig_func, cols=None, **kwargs):
    """Build a table reader for the given format

    Parameters
    ----------
    table : `type`
        table class for which this format is relevant
    format : `str`
        name of the format
    trig_func : `callable`
        method to convert one row of data (from `numpy.loadtxt`) into an event
    cols : `list` of `str`
        list of columns that can be read by default for this format
    **kwargs
        default keyword arguments to pass to `numpy.loadtxt`

    Returns
    -------
    row_reader : `function`
        function that can be used to read a row of this table from ascii.
        The returned function natively supports multi-processing.
    """
    def table_from_ascii_rows(
            f, columns=cols, filt=None, nproc=1, **loadtxtkwargs):
        """Build a `~{0}` from events in an ASCII file.

        Parameters
        ----------
        f : `file`, `str`, `CacheEntry`, `list`, `Cache`
            object representing one or more files. One of

            - an open `file`
            - a `str` pointing to a file path on disk
            - a formatted :class:`~glue.lal.CacheEntry` representing one file
            - a `list` of `str` file paths
            - a formatted :class:`~glue.lal.Cache` representing many files

        columns : `list`, optional
            list of column name strings to read, default all.
        filt : `function`, optional
            function by which to filt events. The callable must accept as
            input a `SnglBurst` event and return `True`/`False`.
        nproc : `int`, optional, default: 1
            number of parallel processes with which to distribute file I/O,
            default: serial process
        **loadtxtkwargs
            all other keyword arguments are passed to `numpy.loadtxt`

        Returns
        -------
        table : `~{0}`
            a new `~{0}` filled with yummy data
        """.format(table.__name__)
        # format keyword arguments
        kwargs_ = kwargs.copy()
        kwargs_.update(loadtxtkwargs)

        # format list of files
        files = file_list(f)

        # allow multiprocessing
        if nproc != 1:
            from ...io.cache import read_cache
            return read_cache(files, table, nproc, return_reassign_ids,
                              columns=columns, format=format, **kwargs_)

        # work out columns to read from ASCII
        if columns is None:
            columns = cols
        else:
            columns = list(columns)
        # and translate them into LIGO_LW columns (only for 'time')
        try:
            ligolwcolumns = list(columns)
        except TypeError as e:
            e.args = ('This ascii format requires the column list to be given '
                      'manually, please give the `columns` keyword argument',)
            raise
        if 'time' in columns and table.tableName in TIME_COLUMN:
            ligolwcolumns.remove('time')
            ligolwcolumns.extend(TIME_COLUMN[table.tableName])

        # generate output
        out = New(table, columns=ligolwcolumns)
        append = out.append

        # get dtypes
        dtype = kwargs_.pop('dtype', None)
        if dtype is None:
            dtype = [(c, 'a20') if c == 'time' else (c, '<f8')
                     for c in columns]

        # iterate over events
        for fp in files:
            dat = loadtxt(fp, dtype=dtype, **kwargs_)
            for line in dat:
                row = trig_func(line, columns=columns)
                if 'event_id' in ligolwcolumns:
                    row.event_id = out.get_next_id()
                if filt is None or filt(row):
                    append(row)
        return out
    return table_from_ascii_rows


def row_from_ascii_factory(table, delimiter):
    """Build a generic ASCII table row reader for the given delimiter
    """
    name = table.tableName
    RowType = table.RowType
    tcols = TIME_COLUMN.get(name, None)

    def row_from_ascii(line, columns):
        """Build a `~{0}` from a line of ASCII data

        Parameters
        ----------
        line : `str`, `array-like`
            a line of ASCII data, either as a delimited string, or an array
        columns : `list` of `str`
            the names of each of the ASCII columns, give 'time' for 'standard'
            time columns for a given table (e.g. 'end_time', and 'end_time_ns'
            for `SnglInspiralTable`)
        delimiter : `str`, optional, default: any whitespace
            the ASCII delimiter, only needed if `line` is given as a `str`

        Returns
        -------
        {1}: `~{0}`
            a new `~{0}` filled with data

        Raises
        ------
        AttributeError
            if any column is not recognised
        """.format(RowType.__name__, StripTableName(name))
        row = table.RowType()
        if isinstance(line, str):
            line = line.rstrip('\n').split(delimiter)
        for datum, colname in zip(line, columns):
            if colname == 'time' and tcols is not None:
                if isinstance(datum, string_types):
                    datum = str(datum)
                t = LIGOTimeGPS(datum)
                setattr(row, tcols[0], t.seconds)
                setattr(row, tcols[1], t.nanoseconds)
            else:
                try:
                    getattr(row, 'set_%s' % colname)(datum)
                except AttributeError:
                    setattr(row, colname, datum)
        return row
    return row_from_ascii

# register generic ASCII parsing for all tables
for table in TableByName.itervalues():
    # register whitespace-delimited ASCII
    register_reader(
        'ascii', table, table_from_ascii_factory(
            table, 'ascii', row_from_ascii_factory(table, None)))
    register_identifier('ascii', table, identify_factory('txt', 'txt.gz'))
    # register csv
    register_reader(
        'csv', table, table_from_ascii_factory(
            table, 'csv', row_from_ascii_factory(table, ','), delimiter=','))
    register_identifier('csv', table, identify_factory('csv', 'csv.gz'))
