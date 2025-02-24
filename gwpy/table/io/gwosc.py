# Copyright (C) Cardiff University (2019-)
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

"""Fetch and parse an event catalog from GWOSC.
"""

from __future__ import annotations

import json
import numbers
import typing

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

from .. import EventTable
from .utils import (
    read_with_columns,
    read_with_where,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import (
        IO,
        Any,
    )

#: suffix indicating a unit name
UNIT_SUFFIX = "_unit"

#: custom GWOSC unit mapping
UNITS: dict[str, units.Quantity] = {
    "M_sun X c^2": units.M_sun * constants.c ** 2,
}

#: set of values corresponding to 'missing' or 'null' data
MISSING_DATA: set[str] = {
    "NA",
}

#: values to fill missing data, based on dtype
_FILL_VALUE: dict[type, Any] = {
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


def _mask_replace(
    value: Any,
    dtype: type,
) -> Any:
    """Replace `value` with the default for the given `dtype`.

    If not default is set for the ``dtype``, just return the
    value unchanged.
    """
    for type_, replacement in _FILL_VALUE.items():
        if issubclass(dtype, type_):
            return replacement
    return value


def _mask_column(col: Iterable) -> tuple[Iterable, list]:
    """Find and replace missing data in a column

    Returns the new data, and the mask as `list`.
    """
    # find masked indices
    mask = [v in MISSING_DATA for v in col]

    # find common dtype of unmasked values
    dtype = numpy.array(x for i, x in enumerate(col) if not mask[i]).dtype.type

    # replace the column with a new version that has the masked
    # values replaced by a 'sensible' default for the relevant dtype
    return (
        [_mask_replace(x, dtype) if mask[i] else x for i, x in enumerate(col)],
        mask,
    )


def fetch_catalog(
    catalog,
    *args,
    host=DEFAULT_GWOSC_URL,
    **kwargs,
):
    """Download a `Table` of events from the GWOSC EventApi.
    """
    # fetch and parse data
    table = parse_eventapi_catalog(
        fetch_catalog_json(catalog, host=host),
        *args,
        **kwargs,
    )

    # include provenance information in metadata
    table.meta.setdefault("catalog", catalog)
    table.meta.setdefault("host", host)

    return table


def read_catalog(
    source: str | Path | IO,
    **kwargs,
):
    """Read a `Table` from a GWOSC EventAPI JSON file.

    Parameters
    ----------
    source : `str`, `~pathlib.Path`, `file`
        A file path or file-like object pointing at GWOSC EventAPI
        JSON data.
    """
    # read a file (object)
    if isinstance(source, str | Path):
        with open(source) as file:
            rawdata = json.load(file)
    else:
        rawdata = json.loads(source)

    return parse_eventapi_catalog(rawdata, **kwargs)


@read_with_columns
@read_with_where
def parse_eventapi_catalog(
    rawdata: dict,
    **kwargs,
):
    """Parse a `Table` from the GWOSC EventAPI JSON output.

    Parameters
    ----------
    rawdata : `dict`
        A blob of JSON data from a `gwosc.api.fetch_catalog_json` request.

    kwargs
        All keyword arguments are passed to the
        `~astropy.table.Table` constructor.

    See also
    --------
    gwosc.api.fetch_catalog_json
        For details of remote data access.

    astropy.table.Table
        For details on how tables are built and what keyword arguments
        are supported.
    """
    data = rawdata["events"]
    coldata: dict[str, list | numpy.ndarray] = {
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

    return tab


EventTable.fetch.registry.register_reader(
    "gwosc",
    EventTable,
    fetch_catalog,
)
