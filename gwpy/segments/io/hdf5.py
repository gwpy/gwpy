# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-)
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

"""This module attaches the HDF5 input output methods to the DataQualityFlag.

While these methods are avialable as methods of the class itself,
this module attaches them to the unified I/O registry, making it a bit
cleaner.
"""

from __future__ import annotations

import os.path
import warnings
from typing import Any

import h5py
import numpy
from astropy.units import (
    Quantity,
    UnitBase,
)

from ...io import hdf5 as io_hdf5
from ...io.registry import (
    register_identifier,
    register_reader,
    register_writer,
)
from ...time import LIGOTimeGPS
from .. import (
    DataQualityDict,
    DataQualityFlag,
    Segment,
    SegmentList,
)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- utilities -----------------------

def find_flag_groups(
    h5group: h5py.Group,
    strict: bool = True,
) -> list[str]:
    """Returns all HDF5 Groups under the given group that contain a flag.

    The check is just that the sub-group has a ``'name'`` attribute, so its
    not fool-proof by any means.

    Parameters
    ----------
    h5group : `h5py.Group`
        the parent group in which to search

    strict : `bool`, optional, default: `True`
        if `True` raise an exception for any sub-group that doesn't have a
        name, otherwise just return all of those that do

    Raises
    ------
    KeyError
        if a sub-group doesn't have a ``'name'`` attribtue and ``strict=True``
    """
    names = []
    for group in h5group:
        try:
            names.append(h5group[group].attrs["name"])
        except KeyError:
            if strict:
                raise
            continue
    return names


# -- read ----------------------------

def _is_flag_group(obj: Any) -> bool:
    """Return `True` if `obj` is an `h5py.Group` that looks like a DQ flag.
    """
    return (
        isinstance(obj, h5py.Group)
        and isinstance(obj.get("active"), h5py.Dataset)
        and isinstance(obj.get("known"), h5py.Dataset)
    )


def _find_flag_groups(h5f: h5py.Group) -> list[str]:
    """Return all groups in `h5f` that look like flags.
    """
    flag_groups: list[str] = []

    def _find(name, obj):
        if _is_flag_group(obj):
            flag_groups.append(name)

    h5f.visititems(_find)
    return flag_groups


def _get_flag_group(
    h5f: h5py.Group,
    path: str | None,
) -> h5py.Group:
    """Determine the group to use in order to read a flag.
    """
    # if user chose the path, just use it
    if path:
        return h5f[path]

    # if the user gave us the group directly, use it
    if _is_flag_group(h5f):
        return h5f

    # otherwise try and find a single group that matches
    try:
        path, = _find_flag_groups(h5f)
    except ValueError:
        pass
    else:
        return h5f[path]

    # if not exactly 1 valid group in the file, complain
    raise ValueError(
        "please pass a valid HDF5 Group, or specify the HDF5 Group "
        "path via the ``path=`` keyword argument",
    )


@io_hdf5.with_read_hdf5
def read_hdf5_flag(
    h5f: h5py.Group,
    path: str | None = None,
    gpstype: type = LIGOTimeGPS,
) -> DataQualityFlag:
    """Read a `DataQualityFlag` object from an HDF5 file or group.
    """
    # extract correct group
    dataset = _get_flag_group(h5f, path)

    # read dataset
    active = SegmentList.read(dataset['active'], format='hdf5',
                              gpstype=gpstype)
    try:
        known = SegmentList.read(dataset['known'], format='hdf5',
                                 gpstype=gpstype)
    except KeyError as first_keyerror:
        try:
            known = SegmentList.read(dataset['valid'], format='hdf5',
                                     gpstype=gpstype)
        except KeyError:
            raise first_keyerror

    return DataQualityFlag(active=active, known=known, **dict(dataset.attrs))


@io_hdf5.with_read_hdf5
def read_hdf5_segmentlist(
    h5f: h5py.Group,
    path: str | None = None,
    gpstype: type = LIGOTimeGPS,
    **kwargs,
) -> SegmentList:
    """Read a `SegmentList` object from an HDF5 file or group.
    """
    from astropy.table import Table

    # find dataset
    dataset = io_hdf5.find_dataset(h5f, path=path)

    segtable = Table.read(dataset, format='hdf5', **kwargs)
    out = SegmentList()
    for row in segtable:
        start = LIGOTimeGPS(int(row['start_time']), int(row['start_time_ns']))
        end = LIGOTimeGPS(int(row['end_time']), int(row['end_time_ns']))
        if gpstype is LIGOTimeGPS:
            out.append(Segment(start, end))
        else:
            out.append(Segment(gpstype(start), gpstype(end)))
    return out


@io_hdf5.with_read_hdf5
def read_hdf5_dict(
    h5f: h5py.Group,
    names: list[str] | None = None,
    path: str | None = None,
    on_missing: str = "error",
    **kwargs,
) -> DataQualityDict:
    """Read a `DataQualityDict` from an HDF5 file.
    """
    if path:
        h5f = h5f[path]

    # try and get list of names automatically
    if names is None:
        try:
            names = find_flag_groups(h5f, strict=True)
        except KeyError:
            names = None
        if not names:
            raise ValueError(
                "Failed to automatically parse available flag names from "
                "HDF5, please give a list of names to read via the "
                "``names=`` keyword",
            )

    # read data
    out = DataQualityDict()
    for name in names:
        try:
            out[name] = read_hdf5_flag(h5f, name, **kwargs)
        except KeyError as exc:
            if on_missing == "ignore":
                continue
            if on_missing == "warn":
                warnings.warn(str(exc))
            else:
                raise ValueError(
                    f"no H5Group found for flag '{name}'",
                )

    return out


# -- write ---------------------------

def write_hdf5_flag_group(
    flag: DataQualityFlag,
    h5group: h5py.Group,
    **kwargs,
) -> h5py.Group:
    """Write a `DataQualityFlag` into the given HDF5 group.
    """
    # write segmentlists
    flag.active.write(h5group, 'active', **kwargs)
    kwargs['append'] = True
    flag.known.write(h5group, 'known', **kwargs)

    # store metadata
    for attr in ['name', 'label', 'category', 'description', 'isgood',
                 'padding']:
        value = getattr(flag, attr)
        if value is None:
            continue
        elif isinstance(value, Quantity):
            h5group.attrs[attr] = value.value
        elif isinstance(value, UnitBase):
            h5group.attrs[attr] = str(value)
        else:
            h5group.attrs[attr] = value

    return h5group


@io_hdf5.with_write_hdf5
def write_hdf5_dict(
    flags: DataQualityDict,
    output: h5py.Group,
    path: str | None = None,
    append: bool = False,
    overwrite: bool = False,
    **kwargs,
):
    """Write this `DataQualityFlag` to a `h5py.Group`.

    This allows writing to an HDF5-format file.

    Parameters
    ----------
    eutput : :class:`h5py.Group`
        Path to new output file, or open h5py `Group` to write to.

    path : `str`
        The HDF5 group path in which to write a new group for this flag

    kwargs
        Other keyword arguments passed to :meth:`h5py.Group.create_dataset`.

    Returns
    -------
    dqfgroup : :class:`h5py.Group`
        HDF group containing these data. This group contains 'active'
        and 'known' datasets, and metadata attrs.

    See also
    --------
    astropy.io
        for details on acceptable keyword arguments when writing a
        :class:`~astropy.table.Table` to HDF5
    """
    if path:
        try:
            parent = output[path]
        except KeyError:
            parent = output.create_group(path)
    else:
        parent = output

    for name in flags:
        # handle existing group
        if name in parent:
            if not (overwrite and append):
                raise OSError(
                    f"Group '{os.path.join(parent.name, name)}' already "
                    "exists, give ``append=True, overwrite=True`` to "
                    "overwrite it",
                )
            del parent[name]
        # create group
        group = parent.create_group(name)
        # write flag
        write_hdf5_flag_group(flags[name], group, **kwargs)


def write_hdf5_flag(
    flag: DataQualityFlag,
    output: h5py.Group,
    path: str | None = None,
    **kwargs,
):
    """Write a `DataQualityFlag` to an HDF5 file/group.
    """
    # verify path (default to flag name)
    if path is None:
        path = flag.name
    if path is None:
        raise ValueError(
            "Cannot determine target group name for flag in HDF5 structure, "
            "please set `name` for each flag, or specify the ``path`` "
            "keyword when writing",
        )

    return write_hdf5_dict({path: flag}, output, **kwargs)


def write_hdf5_segmentlist(
    seglist: SegmentList,
    output: h5py.Group,
    path: str | None = None,
    **kwargs,
):
    """Write a `SegmentList` to an HDF5 file/group.

    Parameters
    ----------
    seglist : :class:`~igwn_segments.segmentlist`
        Data to write.

    output : `str`, `h5py.File`, `h5py.Group`
        Filename or HDF5 object to write to.

    path : `str`
        Path to which to write inside the HDF5 file, relative to ``output``.

    kwargs
        Other keyword arguments are passed to
        :meth:`~astropy.table.Table.write`.
    """
    if path is None:
        raise ValueError(
            "Please specify the HDF5 path via the ``path=`` keyword argument",
        )

    from astropy.table import Table

    # convert segmentlist to Table
    data = numpy.zeros((len(seglist), 4), dtype=int)
    for i, seg in enumerate(seglist):
        start, end = map(LIGOTimeGPS, seg)
        data[i, :] = (
            start.gpsSeconds,
            start.gpsNanoSeconds,
            end.gpsSeconds,
            end.gpsNanoSeconds,
        )
    segtable = Table(
        data,
        names=[
            "start_time",
            "start_time_ns",
            "end_time",
            "end_time_ns",
        ],
    )

    # write table to HDF5
    return segtable.write(
        output,
        path=path,
        format="hdf5",
        **kwargs,
    )


# -- register ------------------------

# SegmentList
register_reader("hdf5", SegmentList, read_hdf5_segmentlist)
register_writer("hdf5", SegmentList, write_hdf5_segmentlist)
register_identifier("hdf5", SegmentList, io_hdf5.identify_hdf5)

# DataQualityFlag
register_reader("hdf5", DataQualityFlag, read_hdf5_flag)
register_writer("hdf5", DataQualityFlag, write_hdf5_flag)
register_identifier(
    "hdf5",
    DataQualityFlag,
    io_hdf5.identify_hdf5,
)

# DataQualityDict
register_reader("hdf5", DataQualityDict, read_hdf5_dict)
register_writer("hdf5", DataQualityDict, write_hdf5_dict)
register_identifier(
    "hdf5",
    DataQualityDict,
    io_hdf5.identify_hdf5,
)
