# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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
from operator import attrgetter

from astropy.units import (Quantity, UnitBase)

from ...detector import Channel
from ...io import (hdf5 as io_hdf5, registry as io_registry)
from ...time import LIGOTimeGPS
from .. import (Array, Series, Index)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

ATTR_TYPE_MAP = {
    Quantity: attrgetter('value'),
    Channel: str,
    UnitBase: str,
    Decimal: float,
    LIGOTimeGPS: float,
}


# -- read ---------------------------------------------------------------------

@io_hdf5.with_read_hdf5
def read_hdf5_array(source, path=None, array_type=Array):
    """Read an `Array` from the given HDF5 object

    Parameters
    ----------
    source : `str`, :class:`h5py.HLObject`
        path to HDF file on disk, or open `h5py.HLObject`.

    path : `str`
        path in HDF hierarchy of dataset.

    array_type : `type`
        desired return type
    """
    dataset = io_hdf5.find_dataset(source, path=path)
    attrs = dict(dataset.attrs)
    # unpickle channel object
    try:
        attrs['channel'] = _unpickle_channel(attrs['channel'])
    except KeyError:  # no channel stored
        pass
    # unpack byte strings for python3
    for key in attrs:
        if isinstance(attrs[key], bytes):
            attrs[key] = attrs[key].decode('utf-8')
    return array_type(dataset[()], **attrs)


def _unpickle_channel(raw):
    """Try and unpickle a channel with sensible error handling
    """
    try:
        return pickle.loads(raw)
    except (ValueError, pickle.UnpicklingError, EOFError, TypeError,
            IndexError) as exc:
        # maybe not pickled
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        try:  # test if this is a valid channel name
            Channel.MATCH.match(raw)
        except ValueError:
            raise exc
        return raw


# -- write --------------------------------------------------------------------

class IgnoredAttribute(ValueError):
    """Internal exception to indicate an attribute to be ignored
    """
    pass


def _format_metadata_attribute(value):
    """Format a value for writing to HDF5 as a `h5py.Dataset` attribute
    """
    if value is None or (isinstance(value, Index) and value.regular):
        raise IgnoredAttribute

    # map type to something HDF5 can handle
    for typekey, func in ATTR_TYPE_MAP.items():
        if issubclass(type(value), typekey):
            return func(value)
    return value


def write_array_metadata(dataset, array):
    """Write metadata for ``array`` into the `h5py.Dataset`
    """
    for attr in ('unit',) + array._metadata_slots:
        # format attribute
        try:
            value = _format_metadata_attribute(
                getattr(array, '_%s' % attr, None))
        except IgnoredAttribute:
            continue

        # store attribute
        try:
            dataset.attrs[attr] = value
        except (TypeError, ValueError, RuntimeError) as exc:
            exc.args = ("Failed to store {} ({}) for {}: {}".format(
                attr, type(value).__name__, type(array).__name__, str(exc)))
            raise


@io_hdf5.with_write_hdf5
def write_hdf5_array(array, h5g, path=None, attrs=None,
                     append=False, overwrite=False,
                     compression='gzip', **kwargs):
    """Write the ``array`` to an `h5py.Dataset`

    Parameters
    ----------
    array : `gwpy.types.Array`
        the data object to write

    h5g : `str`, `h5py.Group`
        a file path to write to, or an `h5py.Group` in which to create
        a new dataset

    path : `str`, optional
        the path inside the group at which to create the new dataset,
        defaults to ``array.name``

    attrs : `dict`, optional
        extra metadata to write into `h5py.Dataset.attrs`, on top of
        the default metadata

    append : `bool`, default: `False`
        if `True`, write new dataset to existing file, otherwise an
        exception will be raised if the output file exists (only used if
        ``f`` is `str`)

    overwrite : `bool`, default: `False`
        if `True`, overwrite an existing dataset in an existing file,
        otherwise an exception will be raised if a dataset exists with
        the given name (only used if ``f`` is `str`)

    compression : `str`, `int`, optional
        compression option to pass to :meth:`h5py.Group.create_dataset`

    **kwargs
        other keyword arguments for :meth:`h5py.Group.create_dataset`

    Returns
    -------
    datasets : `h5py.Dataset`
        the newly created dataset
    """
    if path is None:
        path = array.name
    if path is None:
        raise ValueError("Cannot determine HDF5 path for %s, "
                         "please set ``name`` attribute, or pass ``path=`` "
                         "keyword when writing" % type(array).__name__)

    # create dataset
    dset = io_hdf5.create_dataset(h5g, path, overwrite=overwrite,
                                  data=array.value, compression=compression,
                                  **kwargs)

    # write default metadata
    write_array_metadata(dset, array)

    # allow caller to specify their own metadata dict
    if attrs:
        for key in attrs:
            dset.attrs[key] = attrs[key]

    return dset


def format_index_array_attrs(series):
    """Format metadata attributes for and indexed array

    This function is used to provide the necessary metadata to meet
    the (proposed) LIGO Common Data Format specification for series data
    in HDF5.
    """
    attrs = {}
    # loop through named axes
    for i, axis in zip(range(series.ndim), ('x', 'y')):
        # find property names
        unit = '{}unit'.format(axis)
        origin = '{}0'.format(axis)
        delta = 'd{}'.format(axis)

        # store attributes
        aunit = getattr(series, unit)
        attrs.update({
            unit: str(aunit),
            origin: getattr(series, origin).to(aunit).value,
            delta: getattr(series, delta).to(aunit).value,
        })
    return attrs


def write_hdf5_series(series, output, path=None, attrs=None, **kwargs):
    """Write a Series to HDF5.

    See :func:`write_hdf5_array` for details of arguments and keywords.
    """
    if attrs is None:
        attrs = format_index_array_attrs(series)
    return write_hdf5_array(series, output, path=path, attrs=attrs, **kwargs)


# -- register -----------------------------------------------------------------

def register_hdf5_array_io(array_type, format='hdf5', identify=True):
    """Registry read() and write() methods for the HDF5 format
    """
    def from_hdf5(*args, **kwargs):
        """Read an array from HDF5
        """
        kwargs.setdefault('array_type', array_type)
        return read_hdf5_array(*args, **kwargs)

    io_registry.register_reader(format, array_type, from_hdf5)
    if issubclass(array_type, Series):
        io_registry.register_writer(format, array_type, write_hdf5_series)
    else:
        io_registry.register_writer(format, array_type, write_hdf5_array)

    if identify:
        io_registry.register_identifier(format, array_type,
                                        io_hdf5.identify_hdf5)
