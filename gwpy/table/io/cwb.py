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

"""Read events from Coherent Wave-Burst (cWB)-format ROOT and ASCII files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from astropy.io.ascii import core
from astropy.io.registry import get_reader as astropy_get_reader

from .. import (
    EventTable,
    Table,
)
from .utils import read_with_columns_and_where

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- ROOT ----------------------------

def table_from_cwb_root(
    source: str | Path | IO,
    treename: str = "waveburst",
    **kwargs,
) -> Table:
    """Read an `EventTable` from a Coherent WaveBurst ROOT file.

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


def table_to_cwb_root(
    table: Table,
    target: str | Path | IO,
    treename: str = "waveburst",
    **kwargs,
) -> None:
    """Read an `EventTable` from a Coherent WaveBurst ROOT file.

    This function just redirects to the format='root' reader with appropriate
    defaults.

    See `EventTable.write.help('root')` for more details.
    """
    return table.write(
        target,
        treename=treename,
        **kwargs,
    )


EventTable.read.registry.register_reader(
    "root.cwb",
    EventTable,
    table_from_cwb_root,
)
EventTable.write.registry.register_writer(
    "root.cwb",
    EventTable,
    table_to_cwb_root,
)


# -- ASCII ---------------------------

class CwbHeader(core.BaseHeader):
    """Parser for cWB ASCII header."""

    def get_cols(self, lines: list[str]) -> None:
        """Initialize Column objects from a multi-line ASCII header.

        Parameters
        ----------
        lines : `list`
            List of table lines
        """
        re_name_def = re.compile(
            r"^\s*#\s+"  # whitespace and comment marker
            r"(?P<colnumber>[0-9]+)\s+-\s+"  # number of column
            r"(?P<colname>(.*))",
        )
        self.names = []
        include_cuts = False
        for line in lines:
            if not line:  # ignore empty lines in header (windows)
                continue
            if not line.startswith("# "):  # end of header lines
                break
            if line.startswith("# -/+"):
                include_cuts = True
            else:
                match = re_name_def.search(line)
                if match:
                    self.names.append(match.group("colname").rstrip())

        if not self.names:
            msg = "No column names found in cWB header"
            raise core.InconsistentTableError(msg)

        if include_cuts:
            self.cols = [  # pylint: disable=attribute-defined-outside-init
                core.Column(name="selection cut 1"),
                core.Column(name="selection cut 2"),
            ]
        else:
            self.cols = []  # pylint: disable=attribute-defined-outside-init
        for name in self.names:
            col = core.Column(name=name)
            self.cols.append(col)

    def write(self, lines: list[str]) -> None:
        """Write column headers to ASCII lines."""
        if "selection cut 1" in self.colnames:
            lines.append("# -/+ - not passed/passed final selection cuts")
        for i, name in enumerate(self.colnames):
            lines.append(f"# {i + 1:.2d} - {name}")


class CwbData(core.BaseData):
    """Parser for cWB ASCII data."""

    comment = "#"


class Cwb(core.BaseReader):
    """Read a Cwb file.

    Parameters
    ----------
    Other than the standard keyword options supported for reading ASCII data,
    the following parameters are supported for "ascii.cwb".

    columns : `list` of `str`, optional
        List of column names to read.

    where : `str`, `list` of `str`, optional
        One or more column filters with which to downselect the
        returned table rows as they as read, e.g. ``'snr > 5'``,
        similar to a SQL ``WHERE`` statement.
        Multiple conditions should be connected by ' && ' or ' and ',
        or given as a `list`, e.g. ``'snr > 5 && frequency < 1000'`` or
        ``['snr > 5', 'frequency < 1000']``.
    """

    _format_name = "cwb"
    _io_registry_can_write = True
    _description = "cWB EVENTS format table"

    header_class = CwbHeader
    data_class = CwbData


# register for EventTable
EventTable.read.registry.register_reader(
   "ascii.cwb",
    EventTable,
    read_with_columns_and_where(astropy_get_reader("ascii.cwb", Table)),
)
