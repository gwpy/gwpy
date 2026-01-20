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

"""HDF5 input/output methods for the TimeSeries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy import units

from ...io.hdf5 import (
    identify_hdf5,
    with_read_hdf5,
    with_write_hdf5,
)
from ...types.io.hdf5 import (
    read_hdf5_array,
    write_hdf5_series,
)
from .. import (
    StateVector,
    StateVectorDict,
    TimeSeries,
    TimeSeriesDict,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )

    import h5py

SEC_UNIT = units.second

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- read ----------------------------

def read_hdf5_timeseries(
    h5f: h5py.Group,
    path: str | None = None,
    start: float | None = None,
    end: float | None = None,
    **kwargs,
) -> TimeSeries:
    """Read a `TimeSeries` from HDF5."""
    # read data
    kwargs.setdefault("array_type", TimeSeries)
    series = read_hdf5_array(h5f, path=path, **kwargs)
    # crop if needed
    if start is not None:
        start = max(start, series.span[0])
    if end is not None:
        end = min(end, series.span[1])
    if start is not None or end is not None:
        return series.crop(start, end)
    return series


def _is_timeseries_dataset(dataset: h5py.Dataset) -> bool:
    """Return `True` if a dataset contains `TimeSeries` data."""
    return SEC_UNIT.is_equivalent(dataset.attrs.get("xunit", "undef"))


@with_read_hdf5
def read_hdf5_dict(
    h5f: h5py.Group,
    names: Iterable[str] | None = None,
    group: str | None = None,
    **kwargs,
) -> dict:
    """Read a `TimeSeriesDict` from HDF5."""
    # find group from which to read
    if group:
        h5g = h5f[group]
    else:
        h5g = h5f

    # find list of names to read
    if names is None:
        names = [key for key in h5g if _is_timeseries_dataset(h5g[key])]

    # read names
    out = kwargs.pop("dict_type", TimeSeriesDict)()
    kwargs.setdefault("array_type", out.EntryClass)
    for name in names:
        out[name] = read_hdf5_timeseries(h5g[name], **kwargs)

    return out


def read_hdf5_factory(data_class: type) -> Callable:
    """Create a function to read HDF5 data for the given class type."""
    if issubclass(data_class, dict):
        def read_(*args, **kwargs):  # noqa: ANN002,ANN202
            kwargs.setdefault("dict_type", data_class)
            return read_hdf5_dict(*args, **kwargs)
    else:
        def read_(*args, **kwargs):  # noqa: ANN002,ANN202
            kwargs.setdefault("array_type", data_class)
            return read_hdf5_timeseries(*args, **kwargs)

    return read_


# -- write ---------------------------

@with_write_hdf5
def write_hdf5_dict(
    tsdict: dict,
    h5f: h5py.Group,
    group: str | None = None,
    **kwargs,
) -> None:
    """Write a `TimeSeriesBaseDict` to HDF5.

    Each series in the dict is written as a dataset in the group
    """
    # create group if needed
    if group and group not in h5f:
        h5g = h5f.create_group(group)
    elif group:
        h5g = h5f[group]
    else:
        h5g = h5f

    # write each timeseries
    kwargs.setdefault("format", "hdf5")
    for key, series in tsdict.items():
        series.write(h5g, path=str(key), **kwargs)


# -- register ------------------------

# series classes
for series_class in (TimeSeries, StateVector):
    registry = series_class.read.registry
    reader = read_hdf5_factory(series_class)
    registry.register_reader("hdf5", series_class, reader)
    registry.register_writer("hdf5", series_class, write_hdf5_series)
    registry.register_identifier("hdf5", series_class, identify_hdf5)

# dict classes
for dict_class in (TimeSeriesDict, StateVectorDict):
    registry = series_class.read.registry
    reader = read_hdf5_factory(dict_class)
    registry.register_reader("hdf5", dict_class, reader)
    registry.register_writer("hdf5", dict_class, write_hdf5_dict)
    registry.register_identifier("hdf5", dict_class, identify_hdf5)
