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

"""HDF5 input output methods to the DataQualityFlag.

While these methods are avialable as methods of the class itself,
this module attaches them to the unified I/O registry, making it a bit
cleaner.
"""

# ruff: noqa: D417

from __future__ import annotations

import warnings

import h5py
import numpy
from astropy.table import Table
from astropy.units import (
    Quantity,
    UnitBase,
)

from ...io import hdf5 as io_hdf5
from ...io.registry import default_registry
from ...time import LIGOTimeGPS
from .. import (
    DataQualityDict,
    DataQualityFlag,
    Segment,
    SegmentList,
)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- read ----------------------------

def _is_flag_group(obj: object) -> bool:
    """Return `True` if `obj` is an `h5py.Group` that looks like a DQ flag."""
    return (
        isinstance(obj, h5py.Group)
        and isinstance(obj.get("active"), h5py.Dataset)
        and isinstance(obj.get("known"), h5py.Dataset)
    )


def _find_flag_groups(h5f: h5py.Group) -> list[str]:
    """Return all groups in `h5f` that look like flags."""
    flag_groups: list[str] = []

    def _find(name: str, obj: object) -> None:
        if _is_flag_group(obj):
            flag_groups.append(name)

    h5f.visititems(_find)
    return flag_groups


def _get_flag_group(
    h5f: h5py.Group,
    path: str | None,
) -> h5py.Group:
    """Determine the group to use in order to read a flag."""
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
    msg = (
        "please pass a valid HDF5 Group, or specify the HDF5 Group "
        "path via the ``path=`` keyword argument"
    )
    raise ValueError(msg)


@io_hdf5.with_read_hdf5
def read_hdf5_flag(
    h5f: h5py.Group,
    path: str | None = None,
    gpstype: type = LIGOTimeGPS,
) -> DataQualityFlag:
    """Read a `DataQualityFlag` object from an HDF5 file or group."""
    # extract correct group
    dataset = _get_flag_group(h5f, path)

    # read dataset
    active = SegmentList.read(dataset["active"], format="hdf5", gpstype=gpstype)
    try:
        known = SegmentList.read(dataset["known"], format="hdf5", gpstype=gpstype)
    except KeyError as first_keyerror:
        try:
            known = SegmentList.read(dataset["valid"], format="hdf5", gpstype=gpstype)
        except KeyError:
            raise first_keyerror from None

    return DataQualityFlag(active=active, known=known, **dict(dataset.attrs))


@io_hdf5.with_read_hdf5
def read_hdf5_segmentlist(
    h5f: h5py.Group,
    path: str | None = None,
    gpstype: type = LIGOTimeGPS,
    **kwargs,
) -> SegmentList:
    """Read a `SegmentList` object from an HDF5 file or group.

    Parameters
    ----------
    h5f : `str`, `file`, `h5py.Group`, or `list` of
        File path, open file, or HDF5 group from which to read.

    path : `str`, optional
        Path to the dataset inside the HDF5 file/group. If not given, the
        first dataset found will be used.

    gpstype : `type`, optional
        The class to use for the segment endpoints, default is `LIGOTimeGPS`.

    kwargs
        Other keyword arguments passed to :meth:`~astropy.table.Table.read`.
    """
    # find dataset
    dataset = io_hdf5.find_dataset(h5f, path=path)

    # read table
    segtable = Table.read(dataset, format="hdf5", **kwargs)

    # convert to SegmentList
    out = SegmentList()
    for row in segtable:
        start = LIGOTimeGPS(int(row["start_time"]), int(row["start_time_ns"]))
        end = LIGOTimeGPS(int(row["end_time"]), int(row["end_time_ns"]))
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

    Parameters
    ----------
    h5f : `str`, `file`, `h5py.Group`, or `list` of
        File path, open file, or HDF5 group (or list of any of these)
        from which to read.

    names : `list` of `str`, optional
        List of flag names to read. If not given, will try and find all
        available names automatically.

    path : `str`, optional
        Path to the group inside the HDF5 file/group in which to search
        for flags. If not given, the whole file/group will be searched.

    on_missing : `str`, optional, default: 'error'
        Action to take if a given name is not found, one of:

        - 'error' : raise an exception
        - 'warn' : issue a warning, but continue
        - 'ignore' : silently skip missing names

    kwargs
        Other keyword arguments are passed to
        `DataQualityFlag.read(..., format="hdf5")` to read each flag.

    Raises
    ------
    ValueError
        If no names are given and automatic detection fails, or if
        ``on_missing='error'`` and a given name is not found.

    Returns
    -------
    dqdict : `DataQualityDict`
        A new `DataQualityDict` of `DataQualityFlag` entries with
        ``active`` and ``known`` segments seeded from the HDF5 groups
        in the given file.
    """
    if path:
        h5f = h5f[path]

    # try and get list of names automatically
    if names is None:
        try:
            names = _find_flag_groups(h5f)
        except KeyError:
            names = None
        if not names:
            msg = (
                "Failed to automatically parse available flag names from "
                "HDF5, please give a list of names to read via the "
                "``names=`` keyword"
            )
            raise ValueError(msg)

    # read data
    out = DataQualityDict()
    for name in names:
        try:
            out[name] = read_hdf5_flag(h5f, name, **kwargs)
        except KeyError as exc:
            if on_missing == "ignore":
                continue
            if on_missing == "warn":
                warnings.warn(str(exc), stacklevel=2)
            else:
                msg = f"no H5Group found for flag '{name}'"
                raise ValueError(msg) from None

    return out


# -- write ---------------------------

def write_hdf5_flag_group(
    flag: DataQualityFlag,
    h5group: h5py.Group,
    **kwargs,
) -> h5py.Group:
    """Write a `DataQualityFlag` into the given HDF5 group."""
    # write segmentlists
    flag.active.write(h5group, "active", **kwargs)
    kwargs["append"] = True
    flag.known.write(h5group, "known", **kwargs)

    # store metadata
    for attr in (
        "name",
        "label",
        "category",
        "description",
        "isgood",
        "padding",
    ):
        value = getattr(flag, attr)
        if value is None:
            continue
        if isinstance(value, Quantity):
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
    **kwargs,
) -> None:
    """Write this `DataQualityFlag` to a `h5py.Group`.

    This allows writing to an HDF5-format file.

    Parameters
    ----------
    output : :class:`h5py.Group`
        Path to new output file, or open h5py `Group` to write to.

    path : `str`, optional
        The HDF5 group path in which to write a new group for this flag.
        If not given, the flag name will be used as the group name.

    append : `bool`, optional
        If `False`, and the target file already exists, an
        `OSError` will be raised. If `True`, the file
        will be opened in append mode. Default is `False`.

    overwrite : `bool`, optional
        If `True`, and the target group already exists, the existing
        group will be deleted before writing the new data. Default is
        `False`.

    kwargs
        Other keyword arguments are passed to `SegmentList.write(format="hdf5")`
        for each segment list.

    Returns
    -------
    dqfgroup : :class:`h5py.Group`
        HDF group containing these data. This group contains 'active'
        and 'known' datasets, and metadata attrs.

    See Also
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

    overwrite = kwargs.pop("overwrite", False)

    for name in flags:
        # handle existing group
        if name in parent:
            if not overwrite:
                msg = (
                    f"group '{parent.name}/{name}' already "
                    "exists, give ``overwrite=True`` to "
                    "overwrite it"
                )
                raise OSError(msg)
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
) -> None:
    """Write a `DataQualityFlag` to an HDF5 file/group.

    Parameters
    ----------
    output : `str`, `h5py.File`, `h5py.Group`, or `list` of these
        Filename or HDF5 object (or list of these) to write to.

    path : `str`, optional
        The HDF5 group path in which to write a new group for this flag.
        If not given, the flag name will be used as the group name.

    kwargs
        Other keyword arguments are passed to `SegmentList.write(format="hdf5")`
        for each segment list.
    """
    # verify path (default to flag name)
    if path is None:
        path = flag.name
    if path is None:
        msg = (
            "cannot determine target group name for flag in HDF5 structure, "
            "please set `name` for each flag, or specify the ``path`` "
            "keyword when writing"
        )
        raise ValueError(msg)

    write_hdf5_dict({path: flag}, output, **kwargs)


def write_hdf5_segmentlist(
    seglist: SegmentList,
    output: h5py.Group,
    path: str | None = None,
    **kwargs,
) -> None:
    """Write a `SegmentList` to an HDF5 file/group.

    Parameters
    ----------
    output : `str`, `h5py.File`, `h5py.Group`
        Filename or HDF5 object to write to.

    path : `str`
        Path to which to write inside the HDF5 file, relative to ``output``.

    kwargs
        Other keyword arguments are passed to
        :meth:`~astropy.table.Table.write`.
    """
    if path is None:
        msg = "please specify the HDF5 path via the ``path=`` keyword argument"
        raise ValueError(msg)

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
for klass, read, write in [
    (SegmentList, read_hdf5_segmentlist, write_hdf5_segmentlist),
    (DataQualityFlag, read_hdf5_flag, write_hdf5_flag),
    (DataQualityDict, read_hdf5_dict, write_hdf5_dict),
]:
    default_registry.register_reader("hdf5", klass, read)
    default_registry.register_writer("hdf5", klass, write)
    default_registry.register_identifier("hdf5", klass, io_hdf5.identify_hdf5)
