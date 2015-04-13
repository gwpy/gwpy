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

"""This module provides an extension to the :class:`numpy.ndarray`
data structure providing metadata

The `Array` structure provides the core array-with-metadata environment
with the standard array methods wrapped to return instances of itself.
"""

import warnings
from copy import deepcopy

import numpy
numpy.set_printoptions(threshold=200, linewidth=65)

from astropy.units import (UnitBase, Quantity)
from ..io import (reader, writer)

from .. import version
from ..detector import Channel
from ..time import (Time, to_gps)
from ..utils import with_import

__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Nickolas Fotopoulos <nvf@gravity.phys.uwm.edu>"


# -----------------------------------------------------------------------------
# Core Array

class Array(Quantity):
    """An extension of the :class:`~astropy.units.Quantity`

    This `Array` holds the input data and a standard set of metadata
    properties associated with GW data.

    Parameters
    ----------
    value : array-like, optional, default: `None`
        input data array
    name : `str`, optional, default: `None`
        descriptive title for this `Array`
    unit : `~astropy.units.core.Unit`
        physical unit of these data
    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        starting GPS time of this `Array`, accepts any input for
        :meth:`~gwpy.time.to_gps`.
    channel : `~gwpy.detector.Channel`, `str`
        source data stream for these data
    dtype : :class:`~numpy.dtype`, optional, default: `None`
        input data type
    copy : `bool`, optional, default: `False`
        choose to copy the input data to new memory
    subok : `bool`, optional, default: `True`
        allow passing of sub-classes by the array generator
    **metadata
        other metadata properties

    Returns
    -------
    array : `Array`
        a new array, with a view of the data, and all associated metadata
    """
    _metadata_slots = ['name', 'epoch', 'channel']

    def __new__(cls, value, unit=None, dtype=None, copy=False, subok=True,
                name=None, epoch=None, channel=None):
        """Define a new `Array`, potentially from an existing one
        """
        new = super(Array, cls).__new__(cls, value, dtype=dtype, copy=copy,
                                        subok=subok, unit=unit)
        new.name = name
        new.epoch = epoch
        new.channel = channel
        return new

    # -------------------------------------------
    # array manipulations

    def __quantity_subclass__(self, unit):
        return type(self), True
    __quantity_subclass__.__doc__ = Quantity.__quantity_subclass__.__doc__

    def __array_finalize__(self, obj):
        super(Array, self).__array_finalize__(obj)
        for attr in self._metadata_slots:
            setattr(self, attr, getattr(obj, attr, None))
    __array_finalize__.__doc__ = Quantity.__array_finalize__.__doc__

    def __array_prepare__(self, obj, context=None):
        return super(Array, self).__array_prepare__(obj, context=context)
    __array_prepare__.__doc__ = Quantity.__array_prepare__.__doc__

    def __array_wrap__(self, obj, context=None):
        return super(Array, self).__array_wrap__(obj, context=context)
    __array_wrap__.__doc__ = Quantity.__array_wrap__.__doc__

    def copy(self, order='C'):
        new = super(Array, self).copy(order=order)
        new.__dict__ = deepcopy(self.__dict__)
        return new
    copy.__doc__ = Quantity.copy.__doc__

    def __repr__(self):
        """Return a representation of this object

        This just represents each of the metadata objects appriopriately
        after the core data array
        """
        prefixstr = '<%s(' % self.__class__.__name__
        indent = ' '*len(prefixstr)
        arrstr = numpy.array2string(self.view(numpy.ndarray), separator=',',
                                    prefix=prefixstr)
        metadatarepr = ['unit=%s' % repr(self.unit)]
        for key in self._metadata_slots:
            try:
                val = getattr(self, key)
            except (AttributeError, KeyError):
                val = None
            mindent = ' ' * (len(key) + 1)
            rval = repr(val).replace('\n', '\n%s' % (indent+mindent))
            metadatarepr.append('%s=%s' % (key.strip('_'), rval))
        metadata = (',\n%s' % indent).join(metadatarepr)
        return "{0}{1}\n{2}{3})>".format(
            prefixstr, arrstr, indent, metadata)

    def __str__(self):
        """Return a printable string format representation of this object

        This just prints each of the metadata objects appropriately
        after the core data array
        """
        prefixstr = '%s(' % self.__class__.__name__
        indent = ' '*len(prefixstr)
        arrstr = numpy.array2string(self.view(numpy.ndarray), separator=',',
                                    prefix=prefixstr)
        metadatarepr = ['unit: %s' % repr(self.unit)]
        for key in self._metadata_slots:
            try:
                val = getattr(self, key)
            except (AttributeError, KeyError):
                val = None
            mindent = ' ' * (len(key) + 1)
            rval = repr(val).replace('\n', '\n%s' % (indent+mindent))
            metadatarepr.append('%s: %s' % (key.strip('_'), rval))
        metadata = (',\n%s' % indent).join(metadatarepr)
        return "{0}{1}\n{2}{3})".format(
            prefixstr, arrstr, indent, metadata)

    # -------------------------------------------
    # array methods

    def median(self, axis=None, out=None, overwrite_input=False):
        return numpy.median(self, axis=axis, out=out,
                            overwrite_input=overwrite_input)
    median.__doc__ = numpy.median.__doc__

    @property
    def data(self):
        warnings.warn('Please use \'value\' instead of \'data\' for basic '
                      'numpy.ndarray views', DeprecationWarning)
        return self.value

    # -------------------------------------------
    # Array properties

    @property
    def name(self):
        """Name for this `Array`

        :type: `str`
        """
        return self._name

    @name.setter
    def name(self, val):
        if val is None:
            self._name = None
        else:
            self._name = str(val)

    @property
    def epoch(self):
        """Starting GPS time epoch for this `Array`.

        This attribute is recorded as a `~gwpy.time.Time` object in the
        GPS format, allowing native conversion into other formats.

        See `~astropy.time` for details on the `Time` object.
        """
        if self._epoch is None:
            return None
        else:
            return Time(float(self._epoch),
                        format='gps', scale='utc')

    @epoch.setter
    def epoch(self, epoch):
        if epoch is None:
            self._epoch = None
        else:
            self._epoch = to_gps(epoch)

    @property
    def channel(self):
        """Data channel associated with this `Array`.
        """
        return self._channel

    @channel.setter
    def channel(self, ch):
        if isinstance(ch, Channel):
            self._channel = ch
        elif ch is None:
            self._channel = None
        else:
            self._channel = Channel(ch)

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        raise AttributeError("can't set attribute. To change the units of this"
                             " %s, use the .to() instance method instead."
                             % type(self).__name__)

    # -------------------------------------------
    # extras

    # use input/output registry to allow multi-format reading
    read = classmethod(reader())
    write = writer()

    @with_import('h5py')
    def to_hdf5(self, output, name=None, group=None, compression='gzip',
                **kwargs):
        """Convert this `Array` to a :class:`h5py.Dataset`.

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
            name = name or self.name
            if name is None:
                raise ValueError("Cannot store Array without a name. "
                                 "Either assign the name attribute of the "
                                 "array itself, or given name= as a keyword "
                                 "argument to write().")
            try:
                dset = h5group.create_dataset(
                    name or self.name, data=self.value,
                    compression=compression, **kwargs)
            except ValueError as e:
                if 'Name already exists' in str(e):
                    e.args = (str(e) + ': %s' % (name or self.name),)
                raise

            # store metadata
            for attr in self._metadata_slots:
                mdval = getattr(self, attr)
                if isinstance(mdval, Quantity):
                    dset.attrs[attr] = mdval.value
                elif isinstance(mdval, Channel):
                    dset.attrs[attr] = mdval.ndsname
                elif isinstance(mdval, UnitBase):
                    dset.attrs[attr] = str(mdval)
                elif isinstance(mdval, Time):
                    dset.attrs[attr] = mdval.utc.gps
                else:
                    dset.attrs[attr] = mdval

        finally:
            if not isinstance(output, h5py.Group):
                h5file.close()

        return dset

    @classmethod
    @with_import('h5py')
    def from_hdf5(cls, f, name=None):
        """Read a `Array` from the given HDF file.

        Parameters
        ----------
        f : `str`, :class:`h5py.HLObject`
            path to HDF file on disk, or open `h5py.HLObject`.
        type_ : `type`
            target class to read
        name : `str`
            path in HDF hierarchy of dataset.
        """
        from ..io.hdf5 import open_hdf5

        h5file = open_hdf5(f)

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
                    name2 = '/%s/%s' % (cls.__name__.lower(), name)
                    if name2 in h5file:
                        dataset = h5file[name2]
                    else:
                        raise

            # read array, close file, and return
            out = cls(dataset[()], **dict(dataset.attrs))
        finally:
            if not isinstance(f, (h5py.Dataset, h5py.Group)):
                h5file.close()

        return out
