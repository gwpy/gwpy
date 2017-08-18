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

"""Basic HDF5 I/O methods for Array and sub-classes
"""

import pickle
from decimal import Decimal

from astropy.units import (Quantity, UnitBase)

from ...detector import Channel
from ...io import (hdf5 as io_hdf5, registry as io_registry)
from ...time import (Time, LIGOTimeGPS)
from .. import (Array, Index)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- read ---------------------------------------------------------------------

@io_hdf5.with_read_hdf5
def read_hdf5_array(f, path=None, array_type=Array):
    """Read an `Array` from the given HDF5 object

    Parameters
    ----------
    f : `str`, :class:`h5py.HLObject`
        path to HDF file on disk, or open `h5py.HLObject`.

    path : `str`
        path in HDF hierarchy of dataset.

    array_type : `type`
        desired return type
    """
    dataset = io_hdf5.find_dataset(f, path=path)
    attrs = dict(dataset.attrs)
    try:  # unpickle channel object
        attrs['channel'] = pickle.loads(attrs['channel'])
    except KeyError:  # no channel stored
        pass
    except ValueError:  # not pickled
        pass
    # unpack byte strings for python3
    for key in attrs:
        if isinstance(attrs[key], bytes):
            attrs[key] = attrs[key].decode('utf-8')
    return array_type(dataset[()], **attrs)


# -- write --------------------------------------------------------------------

def create_array_dataset(h5g, array, path=None, compression='gzip', **kwargs):
    """Write the ``array` to an `h5py.Dataset`
    """
    if path is None:
        path = array.name
    if path is None:
        raise ValueError("Cannot determine HDF5 path for %s, "
                         "please set ``name`` attribute, or pass ``path=`` "
                         "keyword when writing" % type(array).__name__)
    try:
        dset = h5g.create_dataset(path, data=array.value,
                                  compression=compression, **kwargs)
    except ValueError as e:
        if 'Name already exists' in str(e):
            e.args = (str(e) + ': %s' % (name or array.name),)
        raise

    # store metadata
    for attr in ('unit',) + array._metadata_slots:
        # get private attribute
        mdval = getattr(array, '_%s' % attr, None)
        if mdval is None:
            continue

        # skip regular index arrays
        if isinstance(mdval, Index) and mdval.regular:
            continue

        # set value based on type
        if isinstance(mdval, Quantity):
            dset.attrs[attr] = mdval.value
        elif isinstance(mdval, Channel):
            dset.attrs[attr] = pickle.dumps(mdval)
        elif isinstance(mdval, UnitBase):
            dset.attrs[attr] = str(mdval)
        elif isinstance(mdval, (Decimal, LIGOTimeGPS)):
            dset.attrs[attr] = str(mdval)
        elif isinstance(mdval, Time):
            dset.attrs[attr] = mdval.utc.gps
        else:
            try:
                dset.attrs[attr] = mdval
            except (TypeError, ValueError, RuntimeError) as e:
                e.args = ("Failed to store %s (%s) for %s: %s"
                          % (attr, type(mdval).__name__,
                             type(array).__name__, str(e)),)
                raise


def write_hdf5_array(array, output, path=None, compression='gzip', **kwargs):
    """Write this array to HDF5
    """
    return io_hdf5.write_object_dataset(array, output, create_array_dataset,
                                        path=path, compression=compression,
                                        **kwargs)


# -- register -----------------------------------------------------------------

def register_hdf5_array_io(array_type, format='hdf5', identify=True):
    """Registry read() and write() methods for the HDF5 format
    """
    def from_hdf5(*args, **kwargs):
        kwargs.setdefault('array_type', array_type)
        return read_hdf5_array(*args, **kwargs)

    def to_hdf5(*args, **kwargs):
        return write_hdf5_array(*args, **kwargs)

    io_registry.register_reader(format, array_type, from_hdf5)
    io_registry.register_writer(format, array_type, to_hdf5)
    if identify:
        io_registry.register_identifier(format, array_type,
                                        io_hdf5.identify_hdf5)
