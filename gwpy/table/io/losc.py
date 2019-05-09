# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019)
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

import numbers
from collections import OrderedDict

import numpy

from astropy import constants
from astropy import units

from gwosc.api import DEFAULT_URL as DEFAULT_GWOSC_URL
from gwosc.catalog import download as download_catalog

from .. import EventTable
from .utils import (
    read_with_columns,
    read_with_selection,
)

#: custom GWOSC unit mapping
UNITS = {
    "M_sun X c^2": units.M_sun * constants.c ** 2,
}

#: set of values corresponding to 'missing' or 'null' data
MISSING_DATA = {
    "NA",
}

#: values to fill missing data, based on dtype
_FILL_VALUE = OrderedDict([
    (str, str()),
    (bytes, bytes()),
    (numbers.Integral, int()),
    (numbers.Number, numpy.nan),
])


__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def _parse_unit(parameter):
    try:
        unit = parameter["unit_en"]
    except KeyError:
        return
    if unit in UNITS:
        return UNITS[unit]
    return unit


def _mask_replace(value, dtype):
    """Replace `value` with the default for the given `dtype`

    If not default is set for the ``dtype``, just return the
    value unchanged.
    """
    for type_, replacement in _FILL_VALUE.items():
        if issubclass(dtype, type_):
            return replacement
    return value


def _mask_column(col):
    """Find and replace missing data in a column

    Returns the new data, and the mask as `list`
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


@read_with_columns
@read_with_selection
def fetch_catalog(catalog, host=DEFAULT_GWOSC_URL):
    catalog = download_catalog(catalog, host=host)
    data = catalog["data"]
    parameters = catalog["parameters"]

    # unpack the catalogue data into a dict of columns
    names = ["name"] + list(parameters.keys())
    cols = {n: [] for n in names}
    for event, plist in data.items():
        cols["name"].append(event)
        for n in names[1:]:
            cols[n].append(plist[n]["best"])

    # rebuild the columns by replacing the masked values
    mask = {}
    for name, col in cols.items():
        cols[name], mask[name] = _mask_column(col)

    # convert to columns
    tab = EventTable(
        cols,
        meta={
            "catalog": catalog,
            "host": host,
        },
        masked=True,
    )

    # add column metadata
    for name, parameter in parameters.items():
        tab[name].mask = mask[name]
        tab[name].description = parameter["name_en"]
        unit = _parse_unit(parameter)
        tab[name].unit = unit

    # add an index on the event name
    tab.add_index('name')

    return tab
