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

"""Read LIGO_LW documents into :class:`~glue.ligolw.table.Table` objects.
"""

import inspect
import warnings

import numpy

try:
    from glue.ligolw.lsctables import TableByName
except ImportError:
    TableByName = dict()

from ...io import registry
from ...io.ligolw import (is_ligolw, read_table as read_ligolw_table,
                          write_tables as write_ligolw_tables)
from .. import (Table, EventTable)
from .utils import read_with_selection

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = []

# methods to exclude from get_as_columns conversions
GET_AS_EXCLUDE = ['get_column', 'get_table']


# -- conversions --------------------------------------------------------------

def to_astropy_table(llwtable, apytable, copy=False, columns=None,
                     rename=None, on_attributeerror=None, get_as_columns=None):
    """Convert a :class:`~glue.ligolw.table.Table` to an `~astropy.tableTable`

    This method is designed as an internal method to be attached to
    :class:`~glue.ligolw.table.Table` objects as `__astropy_table__`.

    Parameters
    ----------
    llwtable : :class:`~glue.ligolw.table.Table`
        the LIGO_LW table to convert from

    apytable : `type`
        `astropy.table.Table` class or subclass

    copy : `bool`, optional
        if `True` copy the input data, otherwise return a reference,
        default: `False`

    columns : `list` of `str`, optional
        the columns to populate, if not given, all columns present in the
        table are mapped

    rename : `dict`, optional
        dict of ('old name', 'new name') pairs to rename columns
        from the original LIGO_LW table

    on_attributeerror
        DEPRECATED, do not use

    get_as_columns
        DEPRECATED, do not use

    Returns
    -------
    table : `EventTable`
        a view of the original data
    """
    # handle deprecated keywords
    if get_as_columns is not None:
        warnings.warn("the ``get_as_columns`` keyword has been deprecated and "
                      "will be removed in an upcoming release, simply refer "
                      "to the columns you want via the ``columns`` keyword",
                      DeprecationWarning)
    if on_attributeerror is not None:
        warnings.warn("the ``on_attributeerror`` keyword has been deprecated, "
                      "and will be removed in an upcoming release, it will be "
                      "ignored for now", DeprecationWarning)

    # set default keywords
    if rename is None:
        rename = {}
    if columns is None:
        columns = llwtable.columnnames

    # get names of get_xxx() methods for this table
    getters = [
        name.split('_', 1)[1] for (name, _) in
        inspect.getmembers(llwtable, predicate=inspect.ismethod) if
        name.startswith('get_') and name not in GET_AS_EXCLUDE and
        name not in llwtable.columnnames]

    # extract columns from LIGO_LW table as astropy.table.Column
    data = []
    for colname in columns:
        # work out how to extract the column
        if colname in getters:
            arr = getattr(llwtable, 'get_{}'.format(colname))()
        else:
            arr = llwtable.getColumnByName(colname)
        # get the data
        copythis = False if colname in getters else copy
        data.append(to_astropy_column(arr, apytable.Column, copy=copythis,
                                      name=rename.get(colname, colname)))

    # build table and return
    return apytable(data, copy=False, meta={'tablename': str(llwtable.Name)})


def to_astropy_column(llwcol, cls, copy=False, dtype=None, **kwargs):
    """Convert a :class:`~glue.ligolw.table.Column` to `astropy.table.Column`

    Parameters
    -----------
    llwcol : :class:`~glue.ligolw.table.Column`, `numpy.ndarray`, iterable
        the LIGO_LW column to convert, or an iterable

    cls : `~astropy.table.Column`
        the Astropy `~astropy.table.Column` or subclass to convert to

    copy : `bool`, optional
        if `True` copy the input data, otherwise return a reference,
        default: `False`

    dtype : `type`, optional
        the data type to convert to when creating the `~astropy.table.Column`

    **kwargs
        other keyword arguments are passed to the `~astropy.table.Column`
        creator

    Returns
    -------
    column : `~astropy.table.Column`
        an Astropy version of the given LIGO_LW column
    """
    if dtype is None:
        dtype = _get_column_dtype(llwcol)
    return cls(data=llwcol, copy=copy, dtype=dtype, **kwargs)


def _get_column_dtype(llwcol):
    """Get the data type of a LIGO_LW `Column`

    Parameters
    ----------
    llwcol : :class:`~glue.ligolw.table.Column`, `numpy.ndarray`, iterable
        a LIGO_LW column, a numpy array, or an iterable

    Returns
    -------
    dtype : `type`, None
        the object data type for values in the given column, `None` is
        returned if ``llwcol`` is a `numpy.ndarray` with `numpy.object_`
        dtype, or no data type can be parsed (e.g. empty list)
    """
    try:  # maybe its a numpy array already!
        dtype = llwcol.dtype
        if dtype is numpy.dtype('O'):  # don't convert
            return None
        return dtype
    except AttributeError:  # dang
        try:  # glue.ligolw.table.Column
            llwtype = llwcol.parentNode.validcolumns[llwcol.Name]
        except AttributeError:  # custom iterator
            try:
                return type(llwcol[0])
            except IndexError:
                return None
        else:
            from glue.ligolw.types import (ToPyType, ToNumPyType)
            try:
                return ToNumPyType[llwtype]
            except KeyError:
                return ToPyType[llwtype]


def table_to_ligolw(table, tablename):
    """Convert a `astropy.table.Table` to a :class:`glue.ligolw.table.Table`
    """
    from glue.ligolw import (lsctables, types)
    from glue.ligolw.ilwd import get_ilwdchar_class

    # create new LIGO_LW table
    columns = table.columns.keys()
    llwtable = lsctables.New(lsctables.TableByName[tablename], columns=columns)

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


# -- read ---------------------------------------------------------------------

@read_with_selection
def read_table(source, tablename=None, **kwargs):
    """Read a `Table` from one or more LIGO_LW XML documents

    source : `file`, `str`, :class:`~glue.ligolw.ligolw.Document`, `list`
        one or more open files, file paths, or LIGO_LW `Document` objects

    tablename : `str`, optional
        the `Name` of the relevant `Table` to read, if not given a table will
        be returned if only one exists in the document(s)

    **kwargs
        keyword arguments for the read, or conversion functions

    See Also
    --------
    gwpy.io.ligolw.read_table
        for details of keyword arguments for the read operation
    gwpy.table.io.ligolw.to_astropy_table
        for details of keyword arguments for the conversion operation
    """
    from glue.ligolw import table as ligolw_table

    # -- keyword handling -----------------------

    # separate keywords for reading and converting from LIGO_LW to Astropy
    read_kw = kwargs  # rename for readability
    convert_kw = {
        'on_attributeerror': None,  # deprecated
        'get_as_columns': None,  # deprecated
        'rename': {},
    }
    for key in convert_kw:
        if key in kwargs:
            convert_kw[key] = kwargs.pop(key)
    if convert_kw['rename'] is None:
        convert_kw['rename'] = {}

    # allow user to specify LIGO_LW columns to read to provide the
    # desired output columns
    try:
        columns = list(kwargs.pop('columns'))
    except KeyError:
        columns = None
    try:
        read_kw['columns'] = list(kwargs.pop('ligolw_columns'))
    except KeyError:
        read_kw['columns'] = columns
    convert_kw['columns'] = columns or read_kw['columns']

    # handle requests for 'time' as a special case - DEPRECATED
    if tablename:
        tableclass = TableByName[ligolw_table.Table.TableName(tablename)]
        needtime = (convert_kw['columns'] is not None and
                    'time' in convert_kw['columns'] and
                    'time' not in tableclass.validcolumns)
        if needtime:
            # find name of real column representing 'time'
            warnings.warn("auto-identification of 'time' column in LIGO_LW "
                          "documents has been deprecated, please directly "
                          "reference the columns you want as written in the "
                          "document(s)", DeprecationWarning)
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
            convert_kw['columns'][convert_kw['columns'].index('time')] = tname
            convert_kw['rename'][tname] = 'time'

            # add required LIGO_LW columns to read_kw
            readcols = [tname, '{}_time'.format(tname),
                        '{}_time_ns'.format(tname)]
            read_kw['columns'] = list(set(read_kw['columns'] + readcols))

    # -- read -----------------------------------

    return Table(read_ligolw_table(source, tablename=tablename, **read_kw),
                 **convert_kw)


# -- write --------------------------------------------------------------------

def write_table(table, target, tablename=None, **kwargs):
    """Write a `~astropy.table.Table` to file in LIGO_LW XML format
    """
    if tablename is None:  # try and get tablename from metadata
        tablename = table.meta.get('tablename', None)
    if tablename is None:  # panic
        raise ValueError("please pass ``tablename=`` to specify the target "
                         "LIGO_LW Table Name")
    return write_ligolw_tables(target, [table_to_ligolw(table, tablename)],
                               **kwargs)


# -- register -----------------------------------------------------------------

for table_class in (Table, EventTable):
    registry.register_reader('ligolw', table_class, read_table)
    registry.register_writer('ligolw', table_class, write_table)
    registry.register_identifier('ligolw', table_class, is_ligolw)


# -- DEPRECATED - remove before 1.0 release -----------------------------------

def _ligolw_io_factory(table):
    """Define a read and write method for the given LIGO_LW table

    This system has been deprecated in favour of the simpler
    `Table.read(format='ligolw', tablename='...')` syntax.
    """
    tablename = table.TableName(table.tableName)
    wng = ("``format='ligolw.{0}'`` has been deprecated, please "
           "read using ``format='ligolw', tablename={0}'``".format(tablename))

    def _read_ligolw_table(source, tablename=tablename, **kwargs):
        warnings.warn(wng, DeprecationWarning)
        return read_table(source, tablename=tablename, **kwargs)

    def _write_table(tbl, target, tablename=tablename, **kwargs):
        warnings.warn(wng, DeprecationWarning)
        return write_table(tbl, target, tablename=tablename, **kwargs)

    return _read_ligolw_table, _write_table


for table_ in TableByName.values():
    # build readers for this table
    read_, write_, = _ligolw_io_factory(table_)

    # register conversion from LIGO_LW to astropy Table
    table_.__astropy_table__ = to_astropy_table

    # register table-specific reader for Table and EventTable
    fmt = 'ligolw.%s' % table_.TableName(table_.tableName)
    registry.register_reader(fmt, Table, read_)
    registry.register_writer(fmt, Table, write_)
