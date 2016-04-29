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

"""Input/output utilities for the `Array`
"""

from astropy.units import Quantity, UnitBase
from astropy.time import Time

from ...detector import Channel
from ...io import (hdf5 as hdf5io, registry)
from ...utils.deps import with_import
from .. import (Array, Series, Array2D)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@with_import('h5py')
def array_from_hdf5(f, name=None, array_type=Array):
    """Read an `Array` from the given HDF5 object

    Parameters
    ----------
    f : `str`, :class:`h5py.HLObject`
        path to HDF file on disk, or open `h5py.HLObject`.

    type_ : `type`
        target class to read

    name : `str`
        path in HDF hierarchy of dataset.
    """
    h5file = hdf5io.open_hdf5(f)

    if name is None and not isinstance(h5file, h5py.Dataset):
        if len(h5file) == 1:
            name = h5file.keys()[0]
        else:
            raise ValueError("Multiple data sets found in HDF structure, "
                             "please give name='...' to specify")

    try:
        # find dataset
        if isinstance(h5file, h5py.Dataset):
            dataset = h5file
        else:
            try:
                dataset = h5file[name]
            except KeyError:
                if name.startswith('/'):
                    raise
                name2 = '/%s/%s' % (array_type.__name__.lower(), name)
                if name2 in h5file:
                    dataset = h5file[name2]
                else:
                    raise

        # read array, close file, and return
        out = array_type(dataset[()], **dict(dataset.attrs))
    finally:
        if not isinstance(f, (h5py.Dataset, h5py.Group)):
            h5file.close()

    return out


@with_import('h5py')
def array_to_hdf5(array, output, name=None, group=None, compression='gzip',
                  array_type=Array, **kwargs):
    """Convert this array to a :class:`h5py.Dataset`.

    This allows writing to an HDF5-format file.

    Parameters
    ----------
    output : `str`, :class:`h5py.Group`
        path to new output file, or open h5py `Group` to write to.

    name : `str`, optional
        custom name for this `Array` in the HDF hierarchy, defaults
        to the `name` attribute of the `Array`.

    group : `str`, optional
        group to create for this time-series.

    compression : `str`, optional
        name of compression filter to use

    **kwargs
        other keyword arguments passed to
        :meth:`h5py.Group.create_dataset`.

    Returns
    -------
    dset : :class:`h5py.Dataset`
        HDF dataset containing these data.
    """
    # create output object
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
        name = name or array.name
        if name is None:
            raise ValueError("Cannot store %s without a name. "
                             "Either assign the name attribute of the "
                             "array itarray, or given name= as a keyword "
                             "argument to write()." % type(array).__name__)
        try:
            dset = h5group.create_dataset(
                name or array.name, data=array.value,
                compression=compression, **kwargs)
        except ValueError as e:
            if 'Name already exists' in str(e):
                e.args = (str(e) + ': %s' % (name or array.name),)
            raise

        # store metadata
        for attr in ['unit'] + array._metadata_slots:
            mdval = getattr(array, attr)
            if mdval is None:
                continue
            if isinstance(mdval, Quantity):
                dset.attrs[attr] = mdval.value
            elif isinstance(mdval, Channel):
                dset.attrs[attr] = mdval.ndsname
            elif isinstance(mdval, UnitBase):
                dset.attrs[attr] = str(mdval)
            elif isinstance(mdval, Time):
                dset.attrs[attr] = mdval.utc.gps
            else:
                try:
                    dset.attrs[attr] = mdval
                except ValueError as e:
                    e.args = ("Failed to store %s (%s) for %s: %s"
                              % (attr, type(mdval).__name__,
                                 type(array).__name__, str(e)),)
                    raise

    finally:
        if not isinstance(output, h5py.Group):
            h5file.close()

    return dset


def register_hdf5_array_io(array_type, format='hdf5', identify=True):
    """Registry read() and write() methods for the HDF5 format
    """
    def from_hdf5(*args, **kwargs):
        kwargs.setdefault('array_type', array_type)
        return array_from_hdf5(*args, **kwargs)
    def to_hdf5(*args, **kwargs):
        kwargs.setdefault('array_type', array_type)
        return array_to_hdf5(*args, **kwargs)
    registry.register_reader(format, array_type, from_hdf5)
    registry.register_writer(format, array_type, to_hdf5)
    if identify:
        registry.register_identifier(format, array_type, hdf5io.identify_hdf5)


# register for basic types
for array_type in (Array, Series, Array2D):
    register_hdf5_array_io(array_type)
    register_hdf5_array_io(array_type, format='hdf', identify=False)
