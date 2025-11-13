# Copyright (c) 2017-2020 Scott Coughlin
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

"""Utilities for database queries.

All functions in this module require :doc:`sqlalchemy <sqlalchemy:index>`.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import (
    TYPE_CHECKING,
    cast,
)

from astropy.table import Table

from .. import EventTable
from ..filter import parse_column_filters

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Mapping,
        Sequence,
    )
    from typing import (
        Any,
        TypeAlias,
    )

    import sqlalchemy

    WhereExpression: TypeAlias = sqlalchemy.SQLColumnExpression | str | None

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- utilities -----------------------

def format_where(
    condition: WhereExpression | Iterable[WhereExpression],
) -> sqlalchemy.SQLColumnExpression | None:
    """Format a column filter condition as a SQL ``WHERE`` expression.

    Requires: :doc:`sqlalchemy <sqlalchemy:index>`

    Parameters
    ----------
    condition : `str`, `list` of `str`
        A column filter string or list of filters.
        Each entry should be parseable using
        `gwpy.table.filter.parse_column_filter`.

    Returns
    -------
    expression : `sqlalchemy.SQLColumnExpression`, `None`
        The formatted query expression, suitable for application using
        `~sqlalchemy.sql.expression.Select.where`.
        `None` will be returned if no ``WHERE`` expression is created
        (``condition`` is `None` or an empty iterable).

    Examples
    --------
    >>> format_where(['snr > 10', 'frequency < 1000'])
    <sqlalchemy.sql.elements.BooleanClauseList object at 0x7f0d7faf78f0>
    >>> print(format_where(['snr > 10', 'frequency < 1000']))
    snr > :snr_1 AND frequency < :frequency_1
    """
    from sqlalchemy import (
        Column,
        SQLColumnExpression,
    )

    # parse condition for SQL query
    if condition is None:
        return None

    if isinstance(condition, str | SQLColumnExpression):
        condition = [condition]

    expressions = []
    for item in condition:
        if item is None:
            continue
        if isinstance(item, SQLColumnExpression):
            expressions.append(item)
        else:
            for col, op_, value in parse_column_filters(item):
                val = cast("SQLColumnExpression", op_(Column(col), value))
                expressions.append(val)
    if expressions:
        return reduce(operator.and_, expressions)
    return None


def format_query(
    tablename: str | sqlalchemy.TableClause,
    columns: Iterable[str | sqlalchemy.Column] | None = None,
    *,
    where: str | list[str] | None = None,
    order_by: str | sqlalchemy.Column | None = None,
    order_by_desc: bool = False,
) -> sqlalchemy.Select:
    """Format a SQL query using `sqlalchemy`.

    Requires: :doc:`sqlalchemy <sqlalchemy:index>`

    Parameters
    ----------
    tablename: `str`
        The name of the database table to ``SELECT FROM``.

    columns : `list` of `str`, optional
        The list of columns to ``SELECT``.

    where : `str`, `list` of `str`, optional
        A filter or list of filters to apply as ``WHERE`` conditions.

    order_by : `str`, optional
        The column to ``ORDER BY``.

    order_by_desc : `bool`, optional
        If `True`, apply ``order_by`` with the ``DESC`` flag.
        Default is `False` (``ASC``).

    Returns
    -------
    statement : `sqlalchemy.sql.expression.Select`
        The formatted ``SELECT`` statement, suitable for passing to
        `sqlalchemy.engine.Connection.execute` or `pandas.read_sql`.
    """
    from sqlalchemy import (
        Column,
        asc,
        desc,
        select,
        table,
    )

    # parse columns
    if columns is None:
        columns = ["*"]
    else:
        columns = list(map(Column, columns))
    columns = cast("list[sqlalchemy.Column]", columns)

    # build SQL query
    if isinstance(tablename, str):
        tablename = table(tablename)
    query = select(*columns).select_from(tablename)
    if (expr := format_where(where)) is not None:
        query = query.where(expr)
    if order_by:
        if order_by_desc:
            return query.order_by(desc(Column(order_by)))
        return query.order_by(asc(Column(order_by)))
    return query


# -- misc queries --------------------

def get_columns(
    tablename: str,
    engine: sqlalchemy.Engine,
) -> dict[str, type]:
    """Return the names and types of the columns in a database table.

    This is just a wrapper around
    `sqlalchemy.engine.reflection.Inspector.get_columns`.

    Parameters
    ----------
    tablename : `str`
        The name of table to inspect.

    engine : `sqlalchemy.engine.Engine`, optional
        The database engine to use when connecting.

    Returns
    -------
    names : `dict` of (`str`, `type`) pairs
        The name and type of each column found in the database.
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)
    return {
        col["name"]: col["type"].python_type
        for col in inspector.get_columns(tablename)
    }


# -- fetch ---------------------------

def create_engine(
    drivername: str,
    username: str | None = None,
    password: str | None = None,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    query: Mapping[str, Sequence[str] | str] | None = None,
    **kwargs,
) -> sqlalchemy.Engine:
    """Create a new `sqlalchemy.engine.Engine`.

    Requires: :doc:`sqlalchemy <sqlalchemy:index>`

    Parameters
    ----------
    drivername : `str`
        Database backend and driver name.

    username : `str`, optional
        The username for authentication to the database.

    password : `str`, optional
        The password for authentication to the database.

    host : `str`, optional
        The name of the remote database host.

    port : `int`, optional
        Port to connect to on ``host``.

    database : `str`, optional
        The name of the database to connect to.

    query : `dict`, optional
        Query parameters.

    kwargs
        Other keyword arguments are passed to `sqlalchemy.create_engine`.

    Returns
    -------
    engine : `sqlalchemy.engine.Engine`
        A new engine.

    See Also
    --------
    sqlalchemy.engine.URL.create
        For details of how URLs are constructed.
    sqlalchemy.create_engine
        For documentation on the engine.
    """
    from sqlalchemy import (
        URL,
        create_engine,
    )

    url = URL.create(
        drivername,
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        query=query or {},
    )
    return create_engine(url, **kwargs)


def fetch(
    # query options
    tablename: str | sqlalchemy.TableClause,
    columns: Iterable[str | sqlalchemy.Column] | None = None,
    *,
    where: str | list[str] | None = None,
    order_by: str | None = None,
    order_by_desc: bool = False,
    # connection options
    engine: sqlalchemy.Engine | None = None,
    drivername: str | None = None,
    username: str | None = None,
    password: str | None = None,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    query: dict[str, Any] | None = None,
    **kwargs,
) -> Table:
    """Fetch data from an SQL table into a `Table`.

    Requires: :doc:`sqlalchemy <sqlalchemy:index>`

    Parameters
    ----------
    tablename : `str`
        The name of table you are attempting to receive triggers
        from.

    columns : `list` of `str`, optional
        The list of columns to ``SELECT``.

    where : `str`, `list` of `str`, optional
        A filter or list of filters to apply as ``WHERE`` conditions,
        e.g. ``'snr > 5'``.
        Multiple conditions should be connected by ' && ', or given as
        a `list`, e.g. ``'snr > 5 && frequency < 1000'`` or
        ``['snr > 5', 'frequency < 1000']``

    order_by : `str`, optional
        The column to ``ORDER BY``.

    order_by_desc : `bool`, optional
        If `True`, apply ``order_by`` with the ``DESC`` flag.
        Default is `False` (``ASC``).

    engine : `sqlalchemy.engine.Engine`, optional
        The database engine to use when connecting.

    drivername : `str`, optional
        Database backend and driver name.
        This is required if ``engine`` is not specified.

    username : `str`, optional
        The username for authentication to the database.

    password : `str`, optional
        The password for authentication to the database.

    host : `str`, optional
        The name of the remote database host.

    port : `int`, optional
        Port to connect to on ``host``.

    database : `str`, optional
        The name of the database to connect to.

    query : `dict`, optional
        Query parameters.

    kwargs
        Other keyword arguments are passed to `pandas.read_sql`.

    Returns
    -------
    table : `Table`
        The table of data received from the SQL query.
    """
    import pandas

    # create engine
    if engine is None:
        if drivername is None:
            msg = "either 'engine' or 'drivername' must be specified"
            raise ValueError(msg)
        engine = create_engine(
            drivername=drivername,
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
            query=query,
        )

    # build query
    qstr = format_query(
        tablename,
        columns=columns,
        where=where,
        order_by=order_by,
        order_by_desc=order_by_desc,
    )

    # kwargs for Table.from_pandas
    index = kwargs.pop("index", False)
    units = kwargs.pop("units", None)

    # perform query
    dataframe = pandas.read_sql(
        qstr,
        engine,
        **kwargs,
    )

    # Convert unicode columns to string
    types = dataframe.apply(lambda x: pandas.api.types.infer_dtype(x.values))
    if not dataframe.empty:
        for col in types[types == "unicode"].index:
            dataframe[col] = dataframe[col].astype(str)

    return Table.from_pandas(
        dataframe,
        index=index,
        units=units,
    ).filled()


EventTable.fetch.registry.register_reader(
    "sql",
    EventTable,
    fetch,
)
