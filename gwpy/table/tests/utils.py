# Copyright (c) 2025 Cardiff University
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

"""Utilties for testing tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy.table import Table
from numpy import dtype
from numpy.random import default_rng

from .. import (
    EventTable,
    GravitySpyTable,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import (
        ArrayLike,
        DTypeLike,
    )

TABLE_CLASSES = (
    Table,
    EventTable,
    GravitySpyTable,
)


def random_table(
    names: Iterable[str],
    length: int = 100,
    table_class: type = EventTable,
    dtypes: list[DTypeLike] | None = None,
) -> Table:
    """Create a table full of random data.

    Parameters
    ----------
    names : `list` of `str`
        The names of the columns to create.

    length : `int`, optional
        The number of rows to create. Default is 100.

    table_class: `type`, optional
        The type of table to create, should be `~astropy.table.Table`
        or a sub-class.
        Default is `~gwpy.table.EventTable`.

    dtypes : `list` of `numpy.dtype`, `type`, `str, optional
        The list of types to associate with each column.
        If given the number of types must match the number of columns
        (``len(dtypes) == len(names)``).
        Defaults to `None` to return `numpy.float64` columns.

    Returns
    -------
    table : instance of ``table_class``
        A table filled with random data.
    """
    names = list(names)
    data: list[ArrayLike | Iterable] = []
    for i in range(len(names)):
        # generate data
        rng = default_rng(seed=i)
        col = rng.random(size=length) * 1000
        # cast to type
        if dtypes:
            dtp = dtypes[i]
            # use map() to support non-primitive types
            if dtype(dtp).name == "object" and callable(dtp):
                data.append(list(map(dtp, col)))
            else:
                data.append(col.astype(dtp))
        else:
            data.append(col)
    return table_class(
        data,
        names=names,
    )
