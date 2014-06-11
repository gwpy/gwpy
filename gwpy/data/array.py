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

from copy import deepcopy

import numpy
numpy.set_printoptions(threshold=200, linewidth=65)

from astropy.units import (Unit, UnitBase, Quantity)
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

class Array(numpy.ndarray):
    """An extension of the :class:`~numpy.ndarray`, with added
    metadata

    This `Array` holds the input data and a standard set of metadata
    properties associated with GW data.

    Parameters
    ----------
    data : array-like, optional, default: `None`
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
    __array_priority_ = 10.1
    _metadata_type = dict
    _metadata_slots = ['name', 'unit', 'epoch', 'channel']

    def __new__(cls, data=None, dtype=None, copy=False, subok=True, **metadata):
        """Define a new `Array`, potentially from an existing one
        """
        # get dtype out of the front
        if isinstance(data, numpy.ndarray) and dtype is None:
            dtype = data.dtype
        elif isinstance(data, numpy.ndarray):
            dtype = numpy.dtype(dtype)

        # copy from an existing Array
        if isinstance(data, cls):
            if not copy and dtype == data.dtype and not metadata:
                return data
            elif metadata:
                new = numpy.array(data, dtype=dtype, copy=copy, subok=True)
                new.metadata = cls._metadata_type(metadata)
                return new
            else:
                new = data.astype(dtype)
                new.metadata = data.metadata
                return new
        # otherwise define a new Array from the array-like data
        else:
            _baseclass = type(data)
            if copy:
                new = super(Array, cls).__new__(cls, data.shape, dtype=dtype)
                try:
                    new[:] = numpy.array(data, dtype=dtype, copy=True)
                except ValueError:
                    pass
            else:
                new = numpy.array(data, dtype=dtype, copy=copy,
                                  subok=True).view(cls)
            new.metadata = cls._metadata_type()
            for key, val in metadata.iteritems():
                if val is not None:
                    setattr(new, key, val)
            new._baseclass = _baseclass
            return new

    # -------------------------------------------
    # array manipulations

    def __array_finalize__(self, obj):
        """Finalize a Array with metadata
        """
        self.metadata = getattr(obj, 'metadata', {}).copy()
        self._baseclass = getattr(obj, '_baseclass', type(obj))

    def __array_wrap__(self, obj, context=None):
        """Wrap an array as a Array with metadata
        """
        result = obj.view(self.__class__)
        result.metadata = self.metadata.copy()
        return result

    def __repr__(self):
        """Return a representation of this object

        This just represents each of the metadata objects appriopriately
        after the core data array
        """
        indent = ' '*len('<%s(' % self.__class__.__name__)
        array = repr(self.data)[6:-1].replace('\n'+' '*6, '\n'+indent)
        if 'dtype' in array:
            array += ','
        metadatarepr = []
        for key in self._metadata_slots:
            try:
                val = getattr(self, key)
            except (AttributeError, KeyError):
                val = None
            mindent = ' ' * (len(key) + 1)
            rval = repr(val).replace('\n', '\n%s' % (indent+mindent))
            metadatarepr.append('%s=%s' % (key, rval))
        metadata = (',\n%s' % indent).join(metadatarepr)
        return "<%s(%s\n%s%s)>" % (self.__class__.__name__, array,
                                   indent, metadata)

    def __str__(self):
        """Return a printable string format representation of this object

        This just prints each of the metadata objects appropriately
        after the core data array
        """
        indent = ' '*len('%s(' % self.__class__.__name__)
        array = str(self.data).replace('\n', '\n' + indent) + ','
        if 'dtype' in array:
            array += ','
        metadatarepr = []
        for key in self._metadata_slots:
            try:
                val = getattr(self, key)
            except (AttributeError, KeyError):
                val = None
            if key == 'epoch' and val is not None:
                val = self.epoch.iso
            elif not val:
                val = None
            mindent = ' ' * (len(key) + 1)
            rval = str(val).replace('\n', '\n%s' % (indent+mindent))
            metadatarepr.append('%s: %s' % (key, rval))
        metadata = (',\n%s' % indent).join(metadatarepr)
        return "%s(%s\n%s%s)" % (self.__class__.__name__, array,
                                 indent, metadata)

    # -------------------------------------------
    # array methods

    def __pow__(self, y, z=None):
        new = self.copy()
        numpy.power(self, y, out=new)
        new.unit = self.unit.__pow__(y)
        return new
    __pow__.__doc__ = numpy.ndarray.__pow__.__doc__

    def __ipow__(self, y):
        super(Array, self).__ipow__(y)
        self.unit **= y
        return self
    __ipow__.__doc__ = numpy.ndarray.__ipow__.__doc__

    def median(self, axis=None, out=None, overwrite_input=False):
        return numpy.median(self, axis=axis, out=out,
                            overwrite_input=overwrite_input)
    median.__doc__ = numpy.median.__doc__

    @property
    def T(self):
        return self.transpose()

    @property
    def H(self):
        return self.T.conj()

    @property
    def data(self):
        return self.view(numpy.ndarray)
    A = data

    def copy(self, order='C'):
        new = super(Array, self).copy(order=order)
        new.metadata = deepcopy(self.metadata)
        return new
    copy.__doc__ = numpy.ndarray.copy.__doc__

    # -------------------------------------------
    # Pickle helpers

    def __getstate__(self):
        """Return the internal state of the object.

        Returns
        -------
        state : `tuple`
            A 5-tuple of (shape, dtype, typecode, rawdata, metadata)
            for pickling
        """
        state = (self.shape,
                 self.dtype,
                 self.flags.fnc,
                 self.data.tostring(),
                 self.metadata
                 )
        return state

    def __setstate__(self, state):
        """Restore the internal state of the `Array`.

        This is used for unpickling purposes.

        Parameters
        ----------
        state : `tuple`
            typically the output of the :meth:`Array.__getstate__`
            method, a 5-tuple containing:

            - class name
            - a tuple giving the shape of the data
            - a typecode for the data
            - a binary string for the data
            - the metadata dict
        """
        (shp, typ, isf, raw, meta) = state
        super(Array, self).__setstate__((shp, typ, isf, raw))
        self.metadata = self._metadata_type(meta)

    def __reduce__(self):
        """Initialise the pickle operation for this `Array`

        Returns
        -------
        pickler : `tuple`
            A 3-tuple of (reconstruct function, reconstruct args, state)
        """
        return (_array_reconstruct, (self.__class__, self.dtype),
                self.__getstate__())

    # -------------------------------------------
    # Array properties

    @property
    def name(self):
        """Name for this `Array`

        :type: `str`
        """
        try:
            return self.metadata['name']
        except KeyError:
            return None

    @name.setter
    def name(self, val):
        self.metadata['name'] = str(val)

    @property
    def unit(self):
        """Unit for this `Array`

        :type: :class:`~astropy.units.core.Unit`
        """
        try:
            return self.metadata['unit']
        except KeyError:
            self.unit = ''
            return self.unit

    @unit.setter
    def unit(self, val):
        if val is None or isinstance(val, Unit):
            self.metadata['unit'] = val
        else:
            self.metadata['unit'] = Unit(val)

    @property
    def epoch(self):
        """Starting GPS time epoch for this `Array`.

        This attribute is recorded as a `~gwpy.time.Time` object in the
        GPS format, allowing native conversion into other formats.

        See `~astropy.time` for details on the `Time` object.
        """
        try:
            return Time(float(self.metadata['epoch']), format='gps', scale='utc')
        except KeyError:
            return None

    @epoch.setter
    def epoch(self, epoch):
        self.metadata['epoch'] = to_gps(epoch)

    @property
    def channel(self):
        """Data channel associated with this `Array`.
        """
        try:
            return self.metadata['channel']
        except KeyError:
            return None

    @channel.setter
    def channel(self, ch):
        if isinstance(ch, Channel):
            self.metadata['channel'] = ch
        else:
            self.metadata['channel'] = Channel(ch)

    # -------------------------------------------
    # extras

    @classmethod
    def _getAttributeNames(cls):
        return cls._metadata_slots

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
            dset = h5group.create_dataset(name or self.name, data=self.data,
                                          compression=compression,
                                          **kwargs)

            # store metadata
            for attr, mdval in self.metadata.iteritems():
                if isinstance(mdval, Quantity):
                    dset.attrs[attr] = mdval.value
                elif isinstance(mdval, Channel):
                    dset.attrs[attr] = mdval.ndsname
                elif isinstance(mdval, UnitBase):
                    dset.attrs[attr] = str(mdval)
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


def _array_reconstruct(class_, dtype):
    """Reconstruct an `Array` after unpickling

    Parameters
    ----------
    Class : `type`, `Array` or sub-class
        class object to create
    dtype : `type`, `numpy.dtype`
        dtype to set
    """
    return class_.__new__(class_, [], dtype=dtype)
