# Copyright (c) 2017-2020 Scott Coughlin
#               2020-2025 Cardiff University
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

"""Input/output methods for tabular data.

Access to GravitySpy and O1GlitchClassification triggers requires access
to a PostgresSQL database. Users can set the username and password for
connections in the following environment variables

- ``GRAVITYSPY_DATABASE_USER``
- ``GRAVITYSPY_DATABASE_PASSWORD``

These can be found https://secrets.ligo.org/secrets/144/. The description
is the username and thesecret is the password.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .. import EventTable
from . import sql as io_sql

if TYPE_CHECKING:
    from collections.abc import (
        Mapping,
        Sequence,
    )

    import sqlalchemy

    from .. import GravitySpyTable

__author__ = "Scott Coughlin <scott.coughlin@ligo.org>"


def get_gravityspy_triggers(
    # query options
    tablename: str | sqlalchemy.TableClause = "glitches",
    columns: list[str | sqlalchemy.Column] | None = None,
    *,
    where: str | list[str] | None = None,
    order_by: str | None = None,
    order_by_desc: bool = False,
    # connection options
    engine: sqlalchemy.Engine | None = None,
    drivername: str = "postgresql",
    username: str | None = None,
    password: str | None = None,
    host: str = "gravityspyplus.ciera.northwestern.edu",
    port: int = 5432,
    database: str = "gravityspy",
    query: Mapping[str, Sequence[str] | str] | None = None,
    **kwargs,
) -> GravitySpyTable:
    """Fetch data into an `GravitySpyTable`.

    Parameters
    ----------
    tablename : `str`,
        The name of table you are attempting to receive triggers from.

    columns : `list` of `str`, optional
        The columns to read from the database.
        If `None` (default) all columns are read.

    where
        other filters you would like to supply
        underlying reader method for the given format

    order_by : `str`, optional
        Column to order results by.

    order_by_desc : `bool`, optional
        Order results in descending order. Default is `False`.

    engine : `sqlalchemy.engine.Engine`, optional
        An existing SQLAlchemy engine to use for the connection.
        If not given, one will be created using the other connection
        parameters.

    drivername : `str`, optional
        Database backend and driver name.
        Default is ``"postgresql"``.

    username : `str`, optional
        The username for authentication to the database.
        Defaults to the value of the ``GRAVITYSPY_DATABASE_USER``
        environment variable.

    password : `str`, optional
        The password for authentication to the database.
        Defaults to the value of the ``GRAVITYSPY_DATABASE_PASSWD``
        environment variable.

    host : `str`, optional
        The name of the server the database you are connecting to
        lives on.
        Default is ``"gravityspyplus.ciera.northwestern.edu"``

    port : `int`, optional
        Port to connect to on ``host``. Default is ``5432``.

    database : `str`, optional
        The name of the SQL database to connect to.
        Default is ``"gravityspy"``.

    query : `dict`, optional
        Additional query parameters used in the database URL.

    kwargs
        Additional keyword arguments are passed to the
        `EventTable.fetch(..., format="sql")`.

    See Also
    --------
    `EventTable.fetch(source='sql')`
        For details of arguments and keyword arguments for this format.

    Returns
    -------
    table : `GravitySpyTable`
    """
    from sqlalchemy.exc import ProgrammingError

    if engine is None:
        engine = create_engine(
            drivername=drivername,
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
            query=query,
        )

    try:
        return EventTable.fetch(
            source="sql",
            tablename=tablename,
            columns=columns,
            where=where,
            order_by=order_by,
            order_by_desc=order_by_desc,
            engine=engine,
            **kwargs,
        )
    except ProgrammingError as exc:
        if f'relation "{tablename}" does not exist' in str(exc):
            tables = _get_table_names(engine)
            msg = exc.args[0]
            msg = msg.replace("does not exist", (
                "does not exist, the following tablenames are "
                "acceptable:\n    {}\n".format(
                    "\n    ".join(tables),
                ),
            ))
            exc.args = (msg,)
        raise


# -- utilities -----------------------

def _get_table_names(engine: sqlalchemy.Engine) -> list[str]:
    """Get a list of table names from a SQLAlchemy engine."""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    return inspector.get_table_names()


def create_engine(
    drivername: str = "postgresql",
    username: str | None = None,
    password: str | None = None,
    host: str = "gravityspyplus.ciera.northwestern.edu",
    port: int = 5432,
    database: str = "gravityspy",
    query: Mapping[str, Sequence[str] | str] | None = None,
    **kwargs,
) -> sqlalchemy.Engine:
    """Create a new `sqlalchemy.engine.Engine` for a GravitySpy query.

    This is just a thin wrapper around `sqlalchemy.engine.URL.create`.

    Parameters
    ----------
    drivername : `str, optional
        Database backend and driver name.
        Default is ``"postgresql"``.

    username : `str`, optional
        The username for authentication to the database.
        Defaults to the value of the ``GRAVITYSPY_DATABASE_USER``
        environment variable.

    password : `str`, optional
        The password for authentication to the database.
        Defaults to the value of the ``GRAVITYSPY_DATABASE_PASSWD``
        environment variable.

    host : `str`, optional
        The name of the server the database you are connecting to
        lives on.
        Default is ``"gravityspyplus.ciera.northwestern.edu"``

    port : `int`, optional
        Port to connect to on ``host``. Default is ``5432``.

    database : `str`, optional
        The name of the SQL database your connecting to.
        Default is ``"gravityspy"``.

    query : `dict`, optional
        Additional query parameters used in the database URL.

    kwargs
        Additional keyword arguments are passed to
        :meth:`gwpy.table.io.sql.create_engine`.

    .. note::

       ``username`` and ``passwd`` should be given together, otherwise they
       will be ignored and values will be resolved from the
       ``GRAVITYSPY_DATABASE_USER`` and ``GRAVITYSPY_DATABASE_PASSWD``
       environment variables.

    Returns
    -------
    engine : `sqlalchemy.engine.Engine`
        A new engine.

    See Also
    --------
    gwpy.table.io.sql.create_engine
        For details of how the engine is created.
    """
    if not username:
        username = os.getenv("GRAVITYSPY_DATABASE_USER")
    if not password:
        password = os.getenv("GRAVITYSPY_DATABASE_PASSWD")

    if not username and not password:
        msg = (
            "username and password not given; "
            "remember to either pass or export GRAVITYSPY_DATABASE_USER "
            "and export GRAVITYSPY_DATABASE_PASSWD in order to access the "
            "Gravity Spy Data; LIGO-Virgo-KAGRA members may visit "
            "https://secrets.ligo.org/secrets/144/ for more information."
        )
        raise ValueError(msg)

    return io_sql.create_engine(
        drivername,
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        query=query,
        **kwargs,
    )


EventTable.fetch.registry.register_reader(
    "gravityspy",
    EventTable,
    get_gravityspy_triggers,
)
