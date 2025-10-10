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

"""Utilities for Table I/O."""

from __future__ import annotations

from functools import wraps
from typing import (
    TYPE_CHECKING,
    cast,
)

from ..filter import filter_table

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Iterable,
        Mapping,
    )
    from typing import (
        ParamSpec,
        TypeVar,
    )

    import h5py
    import numpy
    from astropy.table import Table

    from ..filter import FilterLike

    P = ParamSpec("P")
    T = TypeVar("T", bound=Table)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- dynamic column helpers ----------

def dynamic_columns(
    columns: Iterable[str] | None,
    valid_columns: Collection[str],
    dynamic_column_map: Mapping[str, Iterable[str]],
) -> tuple[set[str] | None, set[str]]:
    """Return the list of columns to read and those to dynamically create.

    Parameters
    ----------
    columns : `list` of `str` or `None`
        The columns requested for the final table.
        If `None` is given, `None` is returned.

    valid_columns : `list` of `str`
        The list of columns that are valid for the table when in the
        relevant format.

    dynamic_column_map: `dict` of `str`, `set` pairs
        The mapping of dynamic column names to columns that need to be read
        to support that column.

    Returns
    -------
    read_columns : `set` of `str`, `None`
        The set of columns to read. Will be `None` if ``columns`` is `None`.

    dynamic_columns : `set` of `str`
        The set of columns to create dynamically after reading.

    Notes
    -----
    The set of column names that is the `~set.difference` between
    the returned ``read_columns`` and ``dynamic_columns`` should be removed
    from the table after the dynamic columns are generated.
    """
    if columns is None:
        return None, set()
    read = set()
    dynamic = set()
    for name in iter(columns):
        # column name is present in the file
        if name in valid_columns:
            read.add(name)
        # column name is a dynamic column that is generated after reading
        elif name in dynamic_column_map:
            dynamic.add(name)
            read.update(dynamic_column_map[name])
        else:
            names = list(valid_columns or []) + list(dynamic_column_map)
            msg = (
                f"'{name}' is not a valid column name; "
                f"valid column names: {', '.join(names)}"
            )
            raise ValueError(msg)
    return read, dynamic


def mtotal(
    table: h5py.Group | Table,
) -> numpy.ndarray:
    """Calculate the total mass column for this table."""
    mass1 = table["mass1"][:]
    mass2 = table["mass2"][:]
    return mass1 + mass2


def mchirp(
    table: h5py.Group | Table,
) -> numpy.ndarray:
    """Calculate the chirp mass column for this table."""
    mass1 = table["mass1"][:]
    mass2 = table["mass2"][:]
    return (mass1 * mass2) ** (3/5.) / (mass1 + mass2) ** (1/5.)


DYNAMIC_COLUMN_FUNC: dict[str, Callable] = {
    "mchirp": mchirp,
    "mtotal": mtotal,
}
DYNAMIC_COLUMN_INPUT: dict[str, set[str]] = {
    "mchirp": {"mass1", "mass2"},
    "mtotal": {"mass1", "mass2"},
}


# -- table i/o utilities -------------

def read_with_columns(
    func: Callable[P, T],
) -> Callable[P, T]:
    """Decorate a Table read method to use the ``columns`` keyword."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # parse columns argument
        columns = kwargs.pop("columns", None)

        # read table
        tab = func(*args, **kwargs)

        # filter on columns
        if columns is None:
            return tab
        try:
            return tab[columns]
        except KeyError as exc:
            missing = str(exc)
            names = list(tab.colnames)
            msg = (
                f"column {missing} not found; "
                f"valid column names: {', '.join(names)}"
            )
            raise ValueError(msg) from exc


    return wrapper


def read_with_where(
    func: Callable[P, T],
) -> Callable[P, T]:
    """Decorate a Table read method to apply the ``where`` keyword.

    Allows for filtering tables on-the-fly when read.
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        """Execute a function, then apply a where filter."""
        # parse where
        where = cast(
            "FilterLike | Iterable[FilterLike]",
            kwargs.pop("where", None) or [],
        )

        # read table
        tab = func(*args, **kwargs)

        # apply where
        if where:
            return filter_table(tab, where)

        return tab

    return wrapper


def read_with_columns_and_where(
    func: Callable[P, T],
) -> Callable[P, T]:
    """Decorate a read function to support both ``columns`` and ``where``.

    The ``where`` decorator is applied _first_ so that the conditions can
    be applied to columns that _aren't_ requested.
    """
    return read_with_columns(read_with_where(func))
