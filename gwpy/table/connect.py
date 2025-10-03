# Copyright (c) 2017 Louisiana State University
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

"""Unified I/O read/write for Table objects."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from astropy.table import vstack
from astropy.table.connect import TableWrite

from ..io.registry import (
    UnifiedFetch,
    UnifiedRead,
    UnifiedWrite,
    default_registry,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from .table import EventTable

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class EventTableRead(UnifiedRead):
    """Read data into an `EventTable`.

    Parameters
    ----------
    source : `str`, `list`
        Source of data, any of the following:

        - `str` path of single data file,
        - `str` path of LAL-format cache file,
        - `list` of paths.

    args
        Other positional arguments will be passed directly to the
        underlying reader method for the given format.

    format : `str`, optional
        The format of the given source files; if not given, an attempt
        will be made to automatically identify the format.

    columns : `list` of `str`, optional
        The list of column names to read.

    where : `str`, or `list` of `str`, optional
        One or more column filters with which to downselect the
        returned table rows as they as read, e.g. ``'snr > 5'``,
        similar to a SQL ``WHERE`` statement.
        Multiple conditions should be connected by ' && ' or ' and ',
        or given as a `list`, e.g. ``'snr > 5 && frequency < 1000'`` or
        ``['snr > 5', 'frequency < 1000']``.

    parallel : `int`
        Number of threads to use for parallel reading of multiple files.

    verbose : `bool`
        Print a progress bar showing read status, default: `False`.

    .. note::

       Keyword arguments other than those listed here may be required
       depending on the `format`

    Returns
    -------
    table : `EventTable`

    Raises
    ------
    astropy.io.registry.IORegistryError
        If the `format` cannot be automatically identified.
    IndexError
        If ``source`` is an empty list.

    Notes
    -----"""

    def merge(  # type: ignore[override]
        self,
        items: Sequence[EventTable],
        join_type: Literal["inner", "exact", "outer"] = "outer",
        metadata_conflicts: Literal["silent", "warn", "error"] = "silent",
    ) -> EventTable:
        """Combine `EventTable` from each file into a single object."""
        return vstack(
            items,
            join_type=join_type,
            metadata_conflicts=metadata_conflicts,
        )

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> EventTable:
        """Read a table."""
        # allow specifying 'where' filter via 'selection' keyword
        if "selection" in kwargs:
            warnings.warn(
                "the 'selection' keyword has been renamed 'where', "
                "this warning will be an error in the near future",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.setdefault("where", kwargs.pop("selection"))

        return super().__call__(
            *args,
            **kwargs,
        )


class EventTableWrite(TableWrite, UnifiedWrite):
    """Write this table to a file in the specified format.

    Get help on the available writers for `EventTable` using the
    ``help()`` method:

      >>> EventTable.write.help()  # general help
      >>> EventTable.write.help('root')  # detailed help for the ROOT writer
      >>> EventTable.write.list_formats()  # print list of available formats

    Parameters
    ----------
    target: `str`
        Filename for output data file.

    *args
        Other positional arguments will be passed directly to the
        underlying writer method for the given format.

    format : `str`
        Format for output data; if not given, an attempt will be made
        to automatically identify the format based on the `target`
        filename.

    **kwargs
        Other keyword arguments will be passed directly to the
        underlying writer method for the given format.

    Raises
    ------
    astropy.io.registry.IORegistryError
        If the `format` cannot be automatically identified.

    Notes
    -----"""

    def __init__(self, instance: EventTable, cls: type[EventTable]) -> None:
        """Create a new `EventTableWrite` instance."""
        UnifiedWrite.__init__(
            self,
            instance,
            cls,
            registry=default_registry,
        )


class EventTableFetch(UnifiedFetch):
    """Fetch a table of events from a database.

    Parameters
    ----------
    *args
        All positional arguments are specific to the
        data source, see below for basic usage.

    source : `str`, `~sqlalchemy.engine.Engine`
        The source of the remote data, see _Notes_ for a list of
        registered sources, OR an SQL database `Engine` object.
        Default is ``'sql'``.

    columns : `list` of `str`, optional
        the columns to fetch from the database table, defaults to all

    where : `str`, or `list` of `str`, optional
        One or more column filters with which to downselect the
        returned table rows as they as read, e.g. ``'snr > 5'``;
        multiple conditions should be connected by ' && ', or given as
        a `list`, e.g. ``'snr > 5 && frequency < 1000'`` or
        ``['snr > 5', 'frequency < 1000']``

    engine : `sqlalchemy.engine.Engine`, optional
        The database engine to use when connecting.

    drivername : `str`
        Database backend and driver name.

    user : `str`, optional
        The username for authentication to the database.

    password : `str`, optional
        The password for authentication to the database.

    host : `str`, optional
        The name of the remote database host.

    post : `int`, optional
        Port to connect to on ``host``.

    database : `str`, optional
        The name of the database to connect to.

    query : `dict`, optional
        Query parameters.

    kwargs
        All other positional arguments are specific to the
        data format, see the online documentation for more details.

    Returns
    -------
    table : `EventTable`
        A table of events recovered from the remote database.

    Examples
    --------
    >>> from gwpy.table import EventTable

    To download a table of all blip glitches from the Gravity Spy database:

    >>> EventTable.fetch(
    ...     tablename="glitches",
    ...     source="gravityspy",
    ...     where=["ml_label=Blip", "ml_confidence>0.9"],
    ... )

    To download a table from any SQL-type server

    >>> EventTable.fetch(
    ...     drivername="postgresql",
    ...     host="localname",
    ...     tablename="data",
    ... )

    Notes
    -----"""

    def __call__(
        self,
        *args,
        source: str = "sql",
        **kwargs,
    ):
        # allow specifying 'where' filter via 'selection' keyword
        if "selection" in kwargs:
            warnings.warn(
                "the 'selection' keyword has been renamed 'where', "
                "this warning will be an error in the near future",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.setdefault("where", kwargs.pop("selection"))

        # if source is given as a SQL Engine, use it
        if hasattr(source, "connect"):
            kwargs.setdefault("engine", source)
            source = "sql"

        return super().__call__(
            *args,
            source=source,
            **kwargs,
        )
