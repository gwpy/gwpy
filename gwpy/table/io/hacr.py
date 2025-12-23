# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

Access to HACR triggers requires local access to the MySQL database. Users
can set the IP address of the server, and the username and password for
connections in the following environment variables

- ``HACR_DATABASE_SERVER``
- ``HACR_DATABASE_USER``
- ``HACR_DATABASE_PASSWD``

These can also be given directly to the connection function as keyword
arguments
"""

from __future__ import annotations

import os.path
from functools import (
    cache,
    partial,
)
from typing import (
    TYPE_CHECKING,
    cast,
)

from astropy.table import (
    Column,
    Table,
    vstack,
)
from dateutil.relativedelta import relativedelta

from ...segments import Segment
from ...time import (
    from_gps,
    to_gps,
)
from .. import EventTable
from . import sql as io_sql
from .utils import dynamic_columns

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    import sqlalchemy

    from ...time import SupportsToGps
    from .sql import WhereExpression

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# get HACR environment variables
HACR_DATABASE_SERVER = os.getenv("HACR_DATABASE_SERVER", None)
HACR_DATABASE_USER = os.getenv("HACR_DATABASE_USER", None)
HACR_DATABASE_QUERY = {
    "charset": "utf8",
}


# -- dynamic columns -----------------

def gpstime(table: Table, name: str = "gpstime") -> Column:
    """Combine the ``gps_start`` and ``gps_offset`` columns."""
    return Column(
        table["gps_start"][:] + table["gps_offset"][:],
        name=name,
    )


DYNAMIC_COLUMN_FUNC: dict[str, Callable] = {
    "gps": partial(gpstime, name="gps"),
    "gpstime": gpstime,
    "time": gpstime,
}
DYNAMIC_COLUMN_INPUT: dict[str, set[str]] = {
    "gps": (_gps_columns := {"gps_start", "gps_offset"}),
    "gpstime": _gps_columns,
    "time": _gps_columns,
}


# -- HACR DB integration -------------

def get_database_names(
    start: SupportsToGps,
    end: SupportsToGps,
) -> list[str]:
    """Return the list of HACR database names covering the given interval.

    Parameters
    ----------
    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    Returns
    -------
    databases : `list` of `str`
        The list of database names that represent the semi-open interval
        ``[start, end)``.

    Examples
    --------
    >>> get_database_names('Jan 1 2024', 'Mar 1 2024')
    ['geo202401', 'geo202402']
    """
    start = int(to_gps(start))
    end = int(to_gps(end))
    if end < start:
        msg = "invalid start/end for HACR query, end must be greater than start"
        raise ValueError(msg)
    if start == end:
        end += 1
    # convert to datetimes
    d = from_gps(start)
    enddate = from_gps(end)
    # loop over months
    dbs = []
    dt = relativedelta(months=1)
    while d < enddate:
        dbs.append(f"geo{d.strftime('%Y%m')}")
        d += dt
    return dbs


def get_hacr_channels(
    gps: SupportsToGps | None = None,
    database: str | None = None,
    engine: sqlalchemy.Engine | None = None,
    **kwargs,
) -> list[str]:
    """Return the names of all channels present in the given HACR database.

    Parameters
    ----------
    gps : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS time of data query, defaults to ```"now"``.
        Any input parseable by `~gwpy.time.to_gps` is fine.

    database : `str`, optional
        The name of the database to query.
        Required if ``engine`` is not given.

    engine : `sqlalchemy.engine.Engine`, optional
        An existing database connection engine.
        If not given, one will be created using the other connection parameters.

    kwargs
        All other keyword arguments are passed to `create_engine`.

    Returns
    -------
    channels : `list` of `str`
        The list of channel names present in the database.

    See Also
    --------
    gwpy.table.io.hacr.create_engine
        For details of valid keyword arguments to create the
        `sqlalchemy.Engine`.
    """
    if engine is None:
        if gps is None:
            gps = to_gps("now") - 43200  # 12 hours ago
        if database is None:
            database, = get_database_names(gps, gps)
        engine = create_engine(
            database,
            **kwargs,
        )
    # query
    with engine.connect() as conn:
        result = conn.execute(
            io_sql.format_query(
                "job",
                columns=["channel"],
                where=["monitorName == 'chacr'"],
            ),
        )
        return list({r.channel for r in result})


def get_hacr_triggers(
    # query options
    channel: str | None = None,
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    columns: Iterable[str] | None = None,
    tablename: str = "mhacr",
    process_id: int | None = None,
    monitor: str = "chacr",
    where: WhereExpression | Iterable[WhereExpression] = None,
    # connection options
    engine: sqlalchemy.Engine | None = None,
    drivername: str = "mysql+pymysql",
    username: str | None = HACR_DATABASE_USER,
    password: str | None = None,
    host: str | None = HACR_DATABASE_SERVER,
    port: int | None = None,
    database: str | None = None,
    query: Mapping[str, Sequence[str] | str] | None = None,
    **kwargs,
) -> Table:
    """Fetch a table of HACR triggers in the given interval.

    Parameters
    ----------
    channel : `str`
        The name of the data channel for which to query.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    columns : `list` of `str`, optional
        The columns to select.
        Default is all columns.

    tablename : `str`, optional
        The name of the database table to select data from.
        Default is ``"mhacr"``.

    process_id : `int`, optional
        The HACR ``process_id`` to match.
        Default is to not select based on ``process_id``.

    monitor : `str`, optional
        The name of the database table to select.
        Default is ``"chacr"``.

    where : `sqlalchemy.sql.elements.SQLColumnExpression`, optional
        A SQLAlchemy expression to use as the ``WHERE`` condition in
        the query.
        Default is `None`.

    engine : `sqlalchemy.engine.Engine`, optional
        An existing database connection engine.
        If not given, one will be created using the other connection parameters.

    drivername : `str`, optional
        The database backend and driver name.
        Default is ``"mysql+pymysql"``.

    username : `str`, optional
        The username for authentication to the database.
        Defaults to the value of the ``HACR_DATABASE_USER`` environment variable.

    password : `str`, optional
        The password for authentication to the database.
        Defaults to the value of the ``HACR_DATABASE_PASSWD`` environment variable.

    host : `str`, optional
        The hostname of the database server.
        Defaults to the value of the ``HACR_DATABASE_SERVER`` environment variable.

    port : `int`, optional
        The port number of the database server.
        Default is the default port for the given ``drivername``.

    database : `str`, optional
        The name of the database to query.
        Required if ``engine`` is not given.

    query : `dict`, optional
        Additional query parameters to use in the database connection.

    kwargs
        All other keyword arguments are passed to `pandas.read_sql`.

    Returns
    -------
    table : `gwpy.table.EventTable`
        A table containing the HACR triggers received from the database.
    """
    # parse given where
    where = [io_sql.format_where(where)]

    # format GPS interval
    if start is not None:
        start = to_gps(start)
    if end is not None:
        end = to_gps(end)

    # get database names to query
    databases: Iterable[str | None]
    if engine is None and database is None:
        if start is None or end is None:
            msg = (
                "start and end options must be given to determine "
                "databases to query"
            )
            raise ValueError(msg)
        databases = get_database_names(start, end)
    else:
        databases = [database]

    # initialise column name holders
    valid_columns: dict[str, type] | None = None
    read_cols: set[str] | None = None
    dynamic_cols: set[str] | None = None

    # loop over databases
    tables: list[EventTable] = []
    for db in databases:
        if engine is None:
            engine = create_engine(
                db,  # type: ignore[arg-type]
                drivername=drivername,
                username=username,
                password=password,
                host=host,
                port=port,
                query=query,
            )

        # format columns
        if valid_columns is None:  # only once
            valid_columns = io_sql.get_columns(
                tablename,
                engine,
            )
            read_cols, dynamic_cols = dynamic_columns(
                columns,
                valid_columns,
                DYNAMIC_COLUMN_INPUT,
            )

        # loop over process_id
        for block in _query_database(
            engine,
            start=start,
            end=end,
            channel=channel,
            columns=read_cols,
            tablename=tablename,
            process_tablename="job",
            monitor=monitor,
            process_id=process_id,
            where=where,
            **kwargs,
        ):
            if len(block):
                tables.append(block)

    if tables:  # join tables we got
        table = vstack(
            tables,
            join_type="exact",
            metadata_conflicts="error",
        )
    elif valid_columns is not None:  # create empty table
        names, dtypes = zip(*(
            (name, valid_columns.get(name))
            for name in read_cols or []
        ), strict=True)
        table = Table(
            names=names,
            dtype=dtypes,
        )
    else:  # no databases found
        msg = "no HACR data found from query"
        raise RuntimeError(msg)

    # add dynamic columns
    if dynamic_cols:
        # if dynamic cols is populated,
        # then columns and read_cols _must_ also be populated
        columns = cast("Iterable[str]", columns)
        read_cols = cast("set[str]", read_cols)
        # generate requested derived columns on-the-fly
        for col_name in dynamic_cols:
            col_data = DYNAMIC_COLUMN_FUNC[col_name](table)
            table.add_column(col_data, name=col_name)
        # remove columns that were only added to generate a derived column
        for col_name in read_cols - set(columns):
            table.remove_column(col_name)

    return table


@cache
def _process_id_query(
    tablename: str,
    monitor: str,
    channel: str | None = None,
) -> sqlalchemy.Select:
    """Construct the ``SELECT`` query to get valid HACR ``process_id``s."""
    where = [
        f"monitorName == '{monitor}'",
    ]
    if channel is not None:
        where.append(f"channel == '{channel}'")
    return io_sql.format_query(
        tablename,
        columns=[
            "process_id",
            "gps_start",
            "gps_stop",
        ],
        where=where,
    )


def _query_database(
    engine: sqlalchemy.Engine,
    start: SupportsToGps | None,
    end: SupportsToGps | None,
    channel: str | None,
    process_id: int | None,
    columns: Iterable[str] | None,
    tablename: str,
    monitor: str,
    process_tablename: str,
    where: Iterable[WhereExpression],
    **kwargs,
) -> Iterator[EventTable]:
    """Query a database for HACR triggers.

    Yields `astropy.table.Table` instances for each ``process_id``.
    """
    # allow querying for all values
    if start is None:
        start = 0
    if end is None:
        end = 1e10

    with engine.connect() as conn:
        if process_id is None:
            # find process ID(s) for this channel
            pids = [
                (row.process_id, row.gps_start, row.gps_stop)
                for row in conn.execute(_process_id_query(
                    process_tablename,
                    monitor,
                    channel=channel,
                ))
            ]
        else:
            pids = [(process_id, start, end)]

        for pid, pid_start, pid_end in pids:
            try:   # get overlap between the given range and the process
                span = Segment(start, end) & Segment(pid_start, pid_end)
            except ValueError:  # no overlap, carry on
                continue
            yield _query_hacr(
                engine,
                start=span[0],
                end=span[1],
                process_id=pid,
                tablename=tablename,
                columns=columns,
                where=where,
                **kwargs,
            )


def _query_hacr(
    engine: sqlalchemy.Engine,
    start: SupportsToGps | None,
    end: SupportsToGps | None,
    process_id: int | None,
    tablename: str,
    columns: Iterable[str] | None,
    where: Iterable[WhereExpression],
    **kwargs,
) -> EventTable:
    """Query for HACR triggers matching the given process_id.

    Returns
    -------
    table : `astropy.table.Table`
        The table of triggers received from the database.
    """
    # recast to append without impacting variable outside scope
    where = list(where)
    # add conditions for this request
    if start is not None:
        where.append(f"gps_start >= {start}")
    if end is not None:
        where.append(f"gps_start < {end}")
    if process_id is not None:
        where.append(f"process_id == {process_id}")

    return EventTable.fetch(
        source="sql",
        engine=engine,
        tablename=tablename,
        columns=columns,
        where=where,
        order_by="gps_start",
        **kwargs,
    )


EventTable.fetch.registry.register_reader(
    "hacr",
    EventTable,
    get_hacr_triggers,
)


# -- utilities -----------------------

def create_engine(
    database: str,
    drivername: str = "mysql+pymysql",
    host: str | None = HACR_DATABASE_SERVER,
    username: str | None = HACR_DATABASE_USER,
    password: str | None = None,
    query: Mapping[str, Sequence[str] | str] | None = None,
    **kwargs,
) -> sqlalchemy.Engine:
    """Create an `sqlalchemy.engine.Engine` for HACR.

    This is just a thin wrapper around `gwpy.table.io.sql.create_engine` with
    HACR defaults.
    """
    if password is None:
        password = os.getenv("HACR_DATABASE_PASSWD") or None
    if query is None:
        query = HACR_DATABASE_QUERY
    return io_sql.create_engine(
        drivername,
        host=host,
        username=username,
        password=password,
        database=database,
        query=query,
        **kwargs,
    )
