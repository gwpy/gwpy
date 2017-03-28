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

import inspect
import warnings

import numpy

import glue.segments
from glue.ligolw.lsctables import (TableByName, LIGOTimeGPS)

from ...io import registry
from ...io.ligolw import (identify_ligolw, table_from_file,
                          write_tables)
from .. import (Table, EventTable)
from ..lsctables import EVENT_TABLES

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = []

INVALID_REC_TYPES = [glue.segments.segment]

# methods to exclude from get_as_columns conversions
GET_AS_EXCLUDE = ['get_column', 'get_table']


# -- read ---------------------------------------------------------------------

def _table_from_ligolw(llwtable, target, copy, columns=None,
                       on_attributeerror='raise', get_as_columns=False,
                       rename={}):
    """Convert this `~glue.ligolw.table.Table` to an `~astropy.tableTable`

        Parameters
        ----------
        columns : `list` of `str`, optional
            the columns to populate, if not given, all columns present in the
            table are mapped

        on_attributeerror : `str`, optional, default: `'raise'`
            how to handle `AttributeError` when accessing rows, one of

            - 'raise' : raise normal exception
            - 'ignore' : silently ignore this column
            - 'warn' : verbosely ignore this column

            .. note::

               this option is ignored if a `columns` list is given

        get_as_columns : `bool`, optional
            convert all `get_xxx()` methods into fields in the
            `~numpy.recarray`; the default is to _not_ do this.

        rename : `dict`, optional
            dict of ('old name', 'new name') pairs to rename columns
            from the original LIGO_LW table

        Returns
        -------
        table : `EventTable`
            a view of the original data
    """
    if rename is None:
        rename = {}
    # create table
    names = []
    data = []

    # and fill it
    for column in llwtable.columnnames:
        if columns and column not in columns:  # skip if not wanted
            continue
        orig_type = llwtable.validcolumns[column]
        try:
            if orig_type == 'ilwd:char':  # numpy tries long() which breaks
                arr = map(int, llwtable.getColumnByName(column))
            else:
                arr = llwtable.getColumnByName(column)
        except AttributeError as e:
            if not columns and on_attributeerror == 'ignore':
                pass
            elif not columns and on_attributeerror == 'warn':
                warnings.warn('Caught %s: %s' % (type(e).__name__, str(e)))
            else:
                raise
        else:
            names.append(column)
            try:
                data.append(target.Column(name=rename[column], data=arr))
            except KeyError:
                data.append(target.Column(name=column, data=arr))

    # fill out get_xxx columns
    if get_as_columns:
        getters = filter(
            lambda x: x[0].startswith('get_') and x[0] not in GET_AS_EXCLUDE,
            inspect.getmembers(llwtable, predicate=inspect.ismethod))
        for name, meth in getters:
            column = name.split('_', 1)[1]
            if column in names:  # don't overwrite existing columns
                continue
            if columns and column not in columns:  # skip if not wanted
                continue
            arr = meth()
            try:
                dtype = arr.dtype
            except AttributeError:
                try:
                    dtype = type(arr[0])
                except (TypeError, KeyError):
                    continue
                except IndexError:
                    dtype = None
            if dtype == LIGOTimeGPS:
                dtype = numpy.float64
            elif dtype in INVALID_REC_TYPES:
                raise TypeError("Cannot store data of type %s in %s"
                                % (dtype, target.__name__))
            names.append(column)
            try:
                data.append(target.Column(name=rename[column], data=arr,
                                          dtype=dtype))
            except KeyError:
                data.append(target.Column(name=column, data=arr, dtype=dtype))

    # sort data columns into user-specified order
    if columns:
        names = [rename[n] if n in rename else n for n in columns]
        data.sort(key=lambda col: names.index(col.name))
    # build table and return
    return target(data, copy=copy,
                  meta={'type': 'ligolw.%s' % str(llwtable.Name)})


# -- write --------------------------------------------------------------------

def table_to_ligolw(table, tablename):
    """Convert a `astropy.table.Table` to a :class:`glue.ligolw.table.Table`
    """
    from glue.ligolw import (lsctables, types)
    from glue.ligolw.ilwd import get_ilwdchar_class

    # create new LIGO_LW table
    columns = table.columns.keys()
    table_class = lsctables.TableByName[tablename]
    llwtable = lsctables.New(table_class, columns=columns)

    # map rows across
    for row in table:
        llwrow = llwtable.RowType()
        for name in columns:
            llwtype = llwtable.validcolumns[name]
            if row[name] is None:
                val = None
            elif llwtype == 'ilwd:char':
                val = get_ilwdchar_class(tablename, name)(row[name])
            else:
                val = types.ToPyType[llwtype](row[name])
            setattr(llwrow, name, val)
        llwtable.append(llwrow)

    return llwtable


# -- I/O factory --------------------------------------------------------------

def ligolw_io_factory(table_):
    """Define a read and write method for the given LIGO_LW table
    """
    tablename = table_.TableName(table_.tableName)

    def _read_ligolw(f, *args, **kwargs):
        return table_from_file(f, tablename, *args, **kwargs)

    def _read_table(f, *args, **kwargs):
        # set up keyword arguments
        llwcolumns = kwargs.pop('ligolw_columns', kwargs.get('columns', None))
        reckwargs = {
            'on_attributeerror': 'raise',
            'get_as_columns': False,
            'rename': {},
        }
        for key in reckwargs:
            if key in kwargs:
                reckwargs[key] = kwargs.pop(key)
        reckwargs['columns'] = kwargs.pop('columns', llwcolumns)
        kwargs['columns'] = llwcolumns
        if reckwargs['rename'] is None:
            reckwargs['rename'] = {}

        # handle requests for 'time' as a special case
        needtime = (reckwargs['columns'] is not None and
                    'time' in reckwargs['columns'] and
                    'time' not in table_.validcolumns)
        if needtime:
            if tablename.endswith('_burst'):
                tname = 'peak'
            elif tablename.endswith('_inspiral'):
                tname = 'end'
            elif tablename.endswith('_ringdown'):
                tname = 'start'
            else:
                raise ValueError("'time' column requested from a table that "
                                 "doesn't supply it or have a good proxy "
                                 "(e.g. 'peak_time')")
            # replace 'time' with get_xxx method name
            reckwargs['columns'] = list(reckwargs['columns'])
            idx = reckwargs['columns'].index('time')
            reckwargs['columns'].insert(idx, tname)
            reckwargs['columns'].pop(idx+1)
            reckwargs['rename'][tname] = 'time'
            reckwargs['get_as_columns'] = True
            # add required LIGO_LW columns to read kwargs
            kwargs['columns'] = list(set(
                kwargs['columns'] + ['%s_time' % tname, '%s_time_ns' % tname]))

        # read from LIGO_LW
        llw = table_from_file(f, table_.tableName, *args, **kwargs)
        return Table(llw, **reckwargs)

    def _write_table(table, f, *args, **kwargs):
        return write_tables(f, [table_to_ligolw(table, tablename)],
                            *args, **kwargs)

    return _read_ligolw, _read_table, _write_table

# -- register -----------------------------------------------------------------

# register reader and auto-id for LIGO_LW
for table in TableByName.values():
    name = 'ligolw.%s' % table.TableName(table.tableName)

    # build readers for this table
    read_llw, read_, write_, = ligolw_io_factory(table)

    # register generic reader and table-specific reader for LIGO_LW
    # DEPRECATED - remove before 1.0 release
    registry.register_reader(name, table, read_llw)
    registry.register_reader('ligolw', table, read_llw)
    registry.register_identifier('ligolw', table, identify_ligolw)

    # register conversion from LIGO_LW to astropy Table
    table.__astropy_table__ = _table_from_ligolw

    # register table-specific reader for Table and EventTable
    registry.register_reader(name, Table, read_)
    registry.register_writer(name, Table, write_)

    if table in EVENT_TABLES:
        # this is done explicitly so that the docstring for table.read()
        # shows the format
        registry.register_reader(name, EventTable, read_)
        registry.register_writer(name, EventTable, write_)
