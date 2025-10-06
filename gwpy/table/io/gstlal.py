# Copyright (c) 2019 Pensylvania State University
#               2019-2022 California Institute of Technology
#               2024-2025 Cardiff University
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

"""Read events from the GstLAL online GW search."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from astropy.table import join

from ...io.ligolw import is_ligolw
from .. import EventTable
from .ligolw import read_table
from .utils import (
    DYNAMIC_COLUMN_FUNC,
    DYNAMIC_COLUMN_INPUT,
    dynamic_columns,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Iterable,
    )
    from typing import (
        IO,
        Literal,
    )

    from astropy.table import (
        Column,
        Table,
    )
    from igwn_ligolw.ligolw import Document

    from ...io.utils import (
        FileLike,
        FileSystemPath,
    )


__author__ = "Derk Davis <derek.davis@ligo.org>"
__credits__ = "Patrick Godwin <patrick.godwin@ligo.org>"

GSTLAL_FORMAT = "ligolw.gstlal"
GSTLAL_SNGL_FORMAT = "ligolw.gstlal.sngl"
GSTLAL_COINC_FORMAT = "ligolw.gstlal.coinc"

GSTLAL_FILENAME = re.compile("([A-Z][0-9])+-LLOID-[0-9.]+-[0-9.]+.xml.gz")


# -- identify ------------------------

def identify_gstlal(
    origin: Literal["read", "write"],
    filepath: FileSystemPath | None,
    fileobj: FileLike | None,
    *args,  # noqa: ANN002
    **kwargs,
) -> bool:
    """Identify a GstLAL file as a ligolw file with the correct name."""
    return (
        is_ligolw(origin, filepath, fileobj, *args, **kwargs)
        and (
            filepath is not None
            and GSTLAL_FILENAME.match(Path(filepath).name) is not None
        )
    )


# -- read ----------------------------

# singles format
def read_gstlal_sngl(
    source: str | Path | IO | Document | list[str | Path | IO],
    columns: Collection[str] | None = None,
    tablename: str = "sngl_inspiral",
    **kwargs,
) -> Table:
    """Read a `sngl_inspiral` table from one or more GstLAL LIGO_LW XML files.

    source : `file`, `str`, `~igwn_ligolw.ligolw.Document`, `list`
        One or more open files or file paths, or a single LIGO_LW ``Document``.

    kwargs
        Other keyword arguments are passed to `Table.read(format="ligolw")`.

    See Also
    --------
    gwpy.io.ligolw.read_table
        For details of keyword arguments for the read operation.

    gwpy.table.io.ligolw.to_astropy_table
        for details of keyword arguments for the conversion operation.
    """
    from igwn_ligolw import lsctables

    # handle column selection
    read_cols, dynamic_cols = dynamic_columns(
        columns,
        lsctables.TableByName[tablename].validcolumns,
        GET_COLUMN_EXTRA,
    )

    # read the table
    events = read_table(
        source,
        tablename=tablename,
        columns=read_cols or None,
        **kwargs,
    )

    if dynamic_cols:
        # generate requested derived columns on-the-fly
        for col_name in dynamic_cols:
            col_data = GET_COLUMN[col_name](events)
            events.add_column(col_data, name=col_name)
        # remove columns that were only added to generate a derived column
        for col_name in (read_cols or set()) - dynamic_cols:
            events.remove_column(col_name)

    return events


# coinc format
def read_gstlal_coinc(
    source: str | Path | IO | Document | list[str | Path | IO],
    columns: Collection[str] | None = None,
    **kwargs,
) -> Table:
    """Read a `Table` containing coinc event info from one or more GstLAL XML files.

    Parameters
    ----------
    source : `file`, `str`, `~igwn_ligolw.ligolw.Document`, `list`
        One or more open files or file paths, or a single LIGO_LW ``Document``.

    columns : `list` of `str`, optional
        List of column names to read. Valid column names are those in
        `coinc_inspiral` and `coinc_event` tables.

    kwargs
        Other keyword arguments are passed to `Table.read(format="ligolw")`.

    See Also
    --------
    gwpy.io.ligolw.read_table
        For details of keyword arguments for the read operation.

    gwpy.table.io.ligolw.to_astropy_table
        for details of keyword arguments for the conversion operation.
    """
    from igwn_ligolw.lsctables import TableByName
    extra_cols = set()

    if columns:
        # if columns were given, separate them into coinc_inspiral and
        # coinc_event columns
        columns = set(columns)
        val_col_inspiral = set(TableByName["coinc_inspiral"].validcolumns)
        val_col_event = set(TableByName["coinc_event"].validcolumns)
        for name in columns:
            if name not in (valid := val_col_inspiral | val_col_event):
                msg = (
                    f"'{name}' is not a valid column name. "
                    f"Valid column names: '{', '.join(valid)}'"
                )
                raise ValueError(msg)
        if "coinc_event_id" not in columns:
            columns.add("coinc_event_id")
            extra_cols.add("coinc_event_id")
        inspiral_cols = columns & val_col_inspiral
        inspiral_cols.add("coinc_event_id")
        event_cols = columns & val_col_event
    else:
        inspiral_cols = None
        event_cols = None

    # read tables
    coinc_inspiral = read_table(
        source,
        tablename="coinc_inspiral",
        columns=inspiral_cols,
        **kwargs,
    )
    coinc_event = read_table(
        source,
        tablename="coinc_event",
        columns=event_cols,
        **kwargs,
    )

    # join events based on coincs
    events = join(
        coinc_inspiral,
        coinc_event,
        keys="coinc_event_id",
        metadata_conflicts="silent",
    )
    events.meta["tablename"] = "gstlal_coinc_inspiral"

    # remove extra columns that we had to read but the user didn't ask for
    for col_name in extra_cols:
        events.remove_column(col_name)

    return events


# combined format
def read_gstlal(
    source: str | Path | IO | Document | list[str | Path | IO],
    triggers: str = "sngl",
    **kwargs,
) -> Table:
    """Read a `Table` from one or more GstLAL LIGO_LW XML files.

    source : `file`, `str`, `~igwn_ligolw.ligolw.Document`, `list`
        One or more open files or file paths, or a single LIGO_LW ``Document``.

    triggers : `str`, optional
        One of

        "sngl"
            For single-detector trigger information.

        "coinc"
            For coincident trigger information.

    kwargs
        Other keyword arguments are passed to `Table.read(format="ligolw")`.

    See Also
    --------
    gwpy.table.io.gstlal.read_gstlal_sngl
    gwpy.table.io.gstlal.read_gstlal_coinc
    """
    if triggers == "sngl":
        return read_gstlal_sngl(source, **kwargs)
    if triggers == "coinc":
        return read_gstlal_coinc(source, **kwargs)
    msg = f"triggers must be 'sngl' or 'coinc', got '{triggers}'"
    raise ValueError(msg)


# registers for unified I/O
EventTable.read.registry.register_identifier(
    GSTLAL_FORMAT,
    EventTable,
    identify_gstlal,
)
for fmt, reader in (
    (GSTLAL_SNGL_FORMAT, read_gstlal_sngl),
    (GSTLAL_COINC_FORMAT, read_gstlal_coinc),
    (GSTLAL_FORMAT, read_gstlal),
):
    EventTable.read.registry.register_reader(fmt, EventTable, reader)

# -- processed columns ---------------
#
# Here we define methods required to build commonly desired columns that
# are just a combination of the basic columns.
#
# Each method should take in a `~gwpy.table.Table` and return a
# `numpy.ndarray` or `~gwpy.table.Column`.

GET_COLUMN: dict[str, Callable] = {}
GET_COLUMN_EXTRA: dict[str, Iterable[str]] = {}


def get_snr_chi(
    events: Table,
    snr_pow: float = 2.,
    chi_pow: float = 2.,
) -> Column:
    """Calculate the 'SNR chi' column for this GstLAL ligolw table group."""
    snr = events["snr"][:]
    chisq = events["chisq"][:]
    return snr**snr_pow / chisq**(chi_pow/2.)


GET_COLUMN["snr_chi"] = get_snr_chi
GET_COLUMN_EXTRA["snr_chi"] = {"snr", "chisq"}


def get_chi_snr(
    events: Table,
    snr_pow: float = 2.,
    chi_pow: float = 2.,
) -> Column:
    """Calculate the 'chi SNR' column for this GstLAL ligolw table group.

    This is the reciprocal of the 'SNR chi' column.
    """
    return 1. / get_snr_chi(events, snr_pow, chi_pow)


GET_COLUMN["chi_snr"] = get_chi_snr
GET_COLUMN_EXTRA["chi_snr"] = {"snr", "chisq"}

# use the generic mass functions
for _key in (
    "mchirp",
    "mtotal",
):
    GET_COLUMN[_key] = DYNAMIC_COLUMN_FUNC[_key]
    GET_COLUMN_EXTRA[_key] = DYNAMIC_COLUMN_INPUT[_key]
