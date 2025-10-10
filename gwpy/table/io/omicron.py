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

"""Read events from an Omicron-format ROOT file."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy.table import Table

from .. import EventTable

if TYPE_CHECKING:
    from ...io.utils import Readable

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def table_from_omicron(
    source: Readable,
    treename: str = "triggers",
    **kwargs,
) -> Table:
    """Read an `EventTable` from an Omicron ROOT file.

    This function just redirects to the format='root' reader with appropriate
    defaults.

    See `EventTable.read.help('root')` for more details.
    """
    return Table.read(
        source,
        format="root",
        treename=treename,
        **kwargs,
    )


for _klass in (Table, EventTable):
    _klass.read.registry.register_reader(
        "root.omicron",
        _klass,
        table_from_omicron,
    )
