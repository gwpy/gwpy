# Copyright (c) 2019-2025 Cardiff University
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

"""Fetch and parse an event catalog from GWOSC."""

from __future__ import annotations

import json
import logging
import numbers
from pathlib import Path
from typing import TYPE_CHECKING

import numpy
from astropy import (
    constants,
    units,
)
from astropy.table import Table
from gwosc.api import (
    DEFAULT_URL as DEFAULT_GWOSC_URL,
    fetch_catalog_json,
)

from ...io.utils import FileSystemPath
from .. import EventTable
from .utils import (
    read_with_columns,
    read_with_where,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...io.utils import Readable

logger = logging.getLogger(__name__)

#: suffix indicating a unit name
UNIT_SUFFIX = "_unit"

#: custom GWOSC unit mapping
UNITS: dict[str, units.Quantity] = {
    "M_sun X c^2": units.M_sun * constants.c ** 2,
}

#: set of values corresponding to 'missing' or 'null' data
MISSING_DATA: set[str | None] = {
    None,
    "NA",
}

#: values to fill missing data, based on dtype
_FILL_VALUE: dict[type, object] = {
    str: "",
    bytes: b"",
    numbers.Integral: 0,
    numbers.Number: numpy.nan,
}

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def _get_unit(
    colname: str,
    unitdict: dict[str, str],
) -> str | units.UnitBase | units.Quantity:
    """Return the unit (name) for a column, or `None` if a match isn't found.

    The return types is either `astropy.units.Unit` or `str`, it doesn't
    matter because setting `column.unit` will automatically convert things
    into `astropy.units.Unit` for us.
    """
    if colname.endswith(("_lower", "_upper")):
        colname = colname.rsplit("_", 1)[0]
    try:
        rawunit = unitdict[colname]
    except KeyError:
        return None
    return UNITS.get(rawunit, rawunit)


def _get_fill_value(type_: type) -> object:
    """Get the fill value for this ``type_``.

    If no default is set for the ``type_``, return `None`.
    """
    for key, replacement in _FILL_VALUE.items():
        if issubclass(type_, key):
            return replacement
    return None


def _mask_replace(
    value: object,
    dtype: type,
) -> object:
    """Replace `value` with the default for the given `dtype`.

    If not default is set for the ``dtype``, just return the
    value unchanged.
    """
    for type_, replacement in _FILL_VALUE.items():
        if issubclass(dtype, type_):
            return replacement
    return value


def _mask_column(col: Iterable) -> tuple[list, list]:
    """Find and replace missing data in a column.

    Returns the new data, and the mask as `list`.
    """
    # find masked indices
    mask = [v in MISSING_DATA for v in col]

    # find common dtype of unmasked values
    dtype = numpy.array([x for i, x in enumerate(col) if not mask[i]]).dtype.type

    # replace the column with a new version that has the masked
    # values replaced by a 'sensible' default for the relevant dtype
    fill_value = _get_fill_value(dtype)
    return (
        [fill_value or x if mask[i] else x for i, x in enumerate(col)],
        mask,
    )


def fetch_catalog(
    catalog: str,
    host: str = DEFAULT_GWOSC_URL,
    **kwargs,
) -> Table:
    """Download a `Table` of events from the GWOSC EventApi."""
    logger.debug("Fetching GWOSC event catalog '%s' from %s", catalog, host)
    # fetch and parse data
    table = parse_eventapi_catalog(
        fetch_catalog_json(catalog, host=host),
        **kwargs,
    )

    # include provenance information in metadata
    table.meta.setdefault("catalog", catalog)
    table.meta.setdefault("host", host)

    return table


def read_catalog(
    source: Readable,
    **kwargs,
) -> Table:
    """Read a `Table` from a GWOSC EventAPI JSON file.

    Parameters
    ----------
    source : `str`, `~pathlib.Path`, `file`
        A file path or file-like object pointing at GWOSC EventAPI
        JSON data.

    kwargs
        Other keyword arguments are passed to `parse_eventapi_catalog`.
    """
    # read a file (object)
    if isinstance(source, FileSystemPath):
        logger.debug("Reading GWOSC event catalog from %s", source)
        with Path(source).open() as file:
            return read_catalog(file)

    # read an open file
    rawdata = json.load(file)

    # and parse it
    return parse_eventapi_catalog(rawdata, **kwargs)


@read_with_columns
@read_with_where
def parse_eventapi_catalog(
    rawdata: dict,
    **kwargs,
) -> Table:
    """Parse a `Table` from the GWOSC EventAPI JSON output.

    Parameters
    ----------
    rawdata : `dict`
        A blob of JSON data from a `gwosc.api.fetch_catalog_json` request.

    kwargs
        All keyword arguments are passed to the
        `~astropy.table.Table` constructor.

    See Also
    --------
    gwosc.api.fetch_catalog_json
        For details of remote data access.

    astropy.table.Table
        For details on how tables are built and what keyword arguments
        are supported.
    """
    data = rawdata["events"]
    coldata: dict[str, list] = {
        "name": list(data.keys()),
    }

    # parse data
    punits = {}
    for event in data.values():
        for key, val in event.items():
            if key.endswith(UNIT_SUFFIX):
                punits[key[:-5]] = val
            else:
                coldata.setdefault(key, []).append(val)

    # rebuild the columns by replacing the masked values
    mask = {}
    for name, col in coldata.items():
        coldata[name], mask[name] = _mask_column(col)

    # convert to columns
    tab = Table(
        coldata,
        masked=True,
        **kwargs,
    )

    # add column metadata
    for name in coldata:
        tab[name].mask = mask[name]
        tab[name].description = name
        tab[name].unit = _get_unit(name, punits)

    # add an index on the event name
    tab.add_index("name")

    logger.debug("Parsed %d events from GWOSC EventAPI", len(tab))
    return tab


EventTable.fetch.registry.register_reader(
    "gwosc",
    EventTable,
    fetch_catalog,
)
