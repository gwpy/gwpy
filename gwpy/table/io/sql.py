# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017-2020)
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

"""Utilities for database queries
"""

from astropy.table import Table

from ..filter import (OPERATORS, parse_column_filters)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def format_db_selection(selection, engine=None):
    """Format a column filter selection as a SQL database WHERE string
    """
    # parse selection for SQL query
    if selection is None:
        return ''
    selections = []
    for col, op_, value in parse_column_filters(selection):
        if engine and engine.name == 'postgresql':
            col = '"%s"' % col
        try:
            opstr = [key for key in OPERATORS if OPERATORS[key] is op_][0]
        except KeyError:
            raise ValueError("Cannot format database 'WHERE' command with "
                             "selection operator %r" % op_)
        selections.append('{0} {1} {2!r}'.format(col, opstr, value))
    if selections:
        return 'WHERE %s' % ' AND '.join(selections)
    return ''


def fetch(engine, tablename, columns=None, selection=None, **kwargs):
    """Fetch data from an SQL table into an `EventTable`

    Parameters
    ----------
    engine : `sqlalchemy.engine.Engine`
        the database engine to use when connecting

    table : `str`,
        The name of table you are attempting to receive triggers
        from.

    selection
        other filters you would like to supply
        underlying reader method for the given format

    .. note::

       For now it will attempt to automatically connect you
       to a specific DB. In the future, this may be an input
       argument.

    Returns
    -------
    table : `GravitySpyTable`
    """
    import pandas as pd

    # parse columns for SQL query
    if columns is None:
        columnstr = '*'
    else:
        columnstr = ', '.join('"%s"' % c for c in columns)

    # parse selection for SQL query
    selectionstr = format_db_selection(selection, engine=engine)

    # build SQL query
    qstr = 'SELECT %s FROM %s %s' % (columnstr, tablename, selectionstr)

    # perform query
    tab = pd.read_sql(qstr, engine, **kwargs)

    # Convert unicode columns to string
    types = tab.apply(lambda x: pd.api.types.infer_dtype(x.values))

    if not tab.empty:
        for col in types[types == 'unicode'].index:
            tab[col] = tab[col].astype(str)

    return Table.from_pandas(tab).filled()
