# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

import h5py
import numpy
from distutils.version import LooseVersion

from astropy.io.registry import (register_reader, register_writer,
                                 register_identifier)
from astropy.units import (UnitBase, Quantity)

from glue.lal import LIGOTimeGPS

from ... import version
from ...io.hdf5 import (open_hdf5, identify_hdf5)
from ..flag import DataQualityFlag
from ..segments import (SegmentList, Segment)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def flag_from_hdf5(f, name=None, gpstype=LIGOTimeGPS, coalesce=True, nproc=1):
    """Read a `DataQualityFlag` object from an HDF5 file or group.
    """
    # hook multiprocessing
    if nproc != 1:
        return DataQualityFlag.read(f, name, coalesce=coalesce,
                                    gpstype=gpstype, format='cache')

    # open file is needed
    h5file = open_hdf5(f)

    try:
        # find dataset
        if name:
            dqfgroup = h5file[name]
        else:
            dqfgroup = h5file

        active = SegmentList.read(dqfgroup['active'])
        valid = SegmentList.read(dqfgroup['valid'])

        # read array, close file, and return
        out = DataQualityFlag(active=active, valid=valid,
                              **dict(dqfgroup.attrs))
    finally:
        if not isinstance(f, (h5py.Dataset, h5py.Group)):
            h5file.close()

    return out


def flag_to_hdf5(flag, output, name=None, group=None, compression='gzip',
                **kwargs):
    """Write this `DataQualityFlag` to a :class:`h5py.Group`.

    This allows writing to an HDF5-format file.

    Parameters
    ----------
    output : `str`, :class:`h5py.Group`
        path to new output file, or open h5py `Group` to write to.
    name : `str`, optional
        custom name for this `Array` in the HDF hierarchy, defaults
        to the `name` attribute of the `Array`.
    group : `str`, optional
        parent group to create for this time-series.
    compression : `str`, optional
        name of compression filter to use
    **kwargs
        other keyword arguments passed to
        :meth:`h5py.Group.create_dataset`.

    Returns
    -------
    dqfgroup : :class:`h5py.Group`
        HDF group containing these data. This group contains 'active'
        and 'valid' datasets, and metadata attrs.
    """
    # create output object
    import h5py
    if isinstance(output, h5py.Group):
        h5file = output
    else:
        h5file = h5py.File(output, 'w')

    try:
        # if group
        if group:
            try:
                h5group = h5file[group]
            except KeyError:
                h5group = h5file.create_group(group)
        else:
            h5group = h5file

        # create dataset
        name = name or flag.name
        if name is None:
            raise ValueError("Cannot store DataQualityFlag without a name. "
                             "Either assign the name attribute of the flag "
                             "itself, or given name= as a keyword argument to "
                             "write().")
        dqfgroup = h5group.create_group(name)

        flag.active.write(dqfgroup, 'active', compression=compression,
                          **kwargs)
        flag.valid.write(dqfgroup, 'valid', compression=compression,
                         **kwargs)

        # store metadata
        for attr in ['name', 'label', 'category', 'description', 'isgood',
                     'padding']:
            value = getattr(flag, attr)
            if value is None:
                continue
            elif isinstance(value, Quantity):
                dqfgroup.attrs[attr] = value.value
            elif isinstance(value, UnitBase):
                dqfgroup.attrs[attr] = str(value)
            else:
                dqfgroup.attrs[attr] = value

    finally:
        if not isinstance(output, (h5py.Dataset, h5py.Group)):
            h5file.close()

    return dqfgroup


def segmentlist_from_hdf5(f, name=None, gpstype=LIGOTimeGPS):
    """Read a `SegmentList` object from an HDF5 file or group.
    """
    h5file = open_hdf5(f)

    try:
        # find dataset
        if isinstance(h5file, h5py.Dataset):
            dataset = h5file
        else:
            dataset = h5file[name]

        try:
            data = dataset[()]
        except ValueError:
            data = []

        out = SegmentList()
        for row in data:
            row = map(int, row)
            # extract as LIGOTimeGPS
            start = LIGOTimeGPS(*row[:2])
            end = LIGOTimeGPS(*row[2:])
            # convert to user type
            try:
                start = gpstype(start)
            except TypeError:
                start = gpstype(float(start))
            try:
                end = gpstype(end)
            except TypeError:
                end = gpstype(float(end))
            out.append(Segment(start, end))
    finally:
        if not isinstance(f, (h5py.Dataset, h5py.Group)):
            h5file.close()

    return out


def segmentlist_to_hdf5(seglist, output, name, group=None,
                        compression='gzip', **kwargs):
    """Write a `SegmentList`
    """
    # create output object
    import h5py
    if isinstance(output, h5py.Group):
        h5file = output
    else:
        h5file = h5py.File(output, 'w')

    try:
        # if group
        if group:
            try:
                h5group = h5file[group]
            except KeyError:
                h5group = h5file.create_group(group)
        else:
            h5group = h5file

        # create dataset
        data = numpy.zeros((len(seglist), 4), dtype=int)
        for i, seg in enumerate(seglist):
            start, end = map(LIGOTimeGPS, seg)
            data[i, :] = (start.seconds, start.nanoseconds,
                          end.seconds, end.nanoseconds)
        if (not len(seglist) and
                LooseVersion(h5py.version.version).version[0] < 2):
            kwargs.setdefault('maxshape', (None, 4))
            kwargs.setdefault('chunks', (1, 1))
        dset = h5group.create_dataset(name, data=data,
                                      compression=compression, **kwargs)
    finally:
        if not isinstance(output, h5py.Group):
            h5file.close()

    return dset


register_reader('hdf', SegmentList, segmentlist_from_hdf5)
register_writer('hdf', SegmentList, segmentlist_to_hdf5)
register_identifier('hdf', SegmentList, identify_hdf5)

register_reader('hdf', DataQualityFlag, flag_from_hdf5)
register_writer('hdf', DataQualityFlag, flag_to_hdf5)
register_identifier('hdf', DataQualityFlag, identify_hdf5)
