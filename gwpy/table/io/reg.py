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

"""Registry utilities for Table I/O."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from .. import EventTable
from .utils import (
    read_with_columns,
    read_with_where,
)

if TYPE_CHECKING:
    from astropy.io.registry import UnifiedIORegistry
    from astropy.table import Table

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def decorate_registered_reader(
    name: str,
    data_class: type[Table] = EventTable,
    *,
    columns: bool = True,
    where: bool = True,
    registry: UnifiedIORegistry | None = None,
) -> None:
    """Wrap an existing registered reader to use GWpy's input decorators.

    Parameters
    ----------
    name : `str`
        The name of the registered format.

    data_class : `type`
        The class for whom the format is registered.

    columns : `bool`
        Use the `read_with_columns` decorator.

    where : `bool`
        Use the `read_with_where` decorator.

    registry : `astropy.io.registry.UnifiedIORegistry`, optional
        The registry to modify.
        If `None` (default) then use the registry for `data_class`.
    """
    if registry is None:
        registry = data_class.read.registry
    orig = reader = registry.get_reader(name, data_class)
    if where:
        reader = wraps(orig)(read_with_where(reader))
    if columns:
        reader = wraps(orig)(read_with_columns(reader))
    return registry.register_reader(
        name,
        data_class,
        reader,
        force=True,
    )


def wrap_unified_io_readers(klass: type[Table]) -> None:
    """Decorate registered readers of ``klass`` to support ``columns`` and ``where``.

    See Also
    --------
    read_with_columns
        For details of the decorator adding ``columns`` keyword support.

    read_with_where
        For details of the decorator adding ``where`` keyword support.
    """
    registry = klass.read.registry
    # wrap each reader to support columns and where keywords
    for row in registry.get_formats(
        data_class=klass,
        readwrite="Read",
    ):
        decorate_registered_reader(
            row["Format"],
            data_class=klass,
            columns=True,
            where=True,
            registry=registry,
        )
