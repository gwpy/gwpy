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

from astropy.units import Quantity
from ..io import (reader, writer)

from ..detector import Channel
from ..detector.units import parse_unit
from ..time import (Time, to_gps)
from ..utils.docstring import interpolate_docstring

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Nickolas Fotopoulos <nvf@gravity.phys.uwm.edu>"

numpy.set_printoptions(threshold=200, linewidth=65)

# -----------------------------------------------------------------------------
# Core Array

# update docstring interpreter with generic Array parameters
interpolate_docstring.update(
    Array1="""value : array-like
        input data array

    unit : `~astropy.units.Unit`, optional
        physical unit of these data

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine""",

    Array2="""name : `str`, optional, default: `None`
        descriptive title for this array

    channel : `~gwpy.detector.Channel`, `str`
        source data stream for these data

    dtype : :class:`~numpy.dtype`, optional, default: `None`
        input data type

    copy : `bool`, optional, default: `False`
        choose to copy the input data to new memory

    subok : `bool`, optional, default: `True`
        allow passing of sub-classes by the array generator""",
)


@interpolate_docstring
class Array(Quantity):
    """An extension of the :class:`~astropy.units.Quantity`

    This `Array` holds the input data and a standard set of metadata
    properties associated with GW data.

    Parameters
    ----------
    %(Array1)s

    %(Array2)s

    Returns
    -------
    array : `Array`
        a new array, with a view of the data, and all associated metadata

    Examples
    --------
    To create a new `Array` from a list of samples:

        >>> a = Array([1, 2, 3, 4, 5], 'm/s', name='my data')
        >>> print(a)
        Array([ 1., 2., 3., 4., 5.]
              unit: Unit("m / s"),
              name: 'my data',
              epoch: None,
              channel: None)

    """
    _metadata_slots = ['name', 'epoch', 'channel']

    def __new__(cls, value, unit=None, dtype=None, copy=False, subok=True,
                order=None, name=None, epoch=None, channel=None):
        """Define a new `Array`, potentially from an existing one
        """
        if dtype is None and isinstance(value, numpy.ndarray):
            dtype = value.dtype

        unit = parse_unit(unit, parse_strict='warn')
        new = super(Array, cls).__new__(cls, value, dtype=dtype, copy=copy,
                                        subok=subok, order=order, unit=unit)
        new.name = name
        new.epoch = epoch
        new.channel = channel
        return new

    # -------------------------------------------
    # array manipulations

    def _wrap_function(self, function, *args, **kwargs):
        out = super(Array, self)._wrap_function(function, *args, **kwargs)
        if out.ndim == 0:
            return Quantity(out.value, out.unit)
        return out

    def __quantity_subclass__(self, unit):
        return type(self), True

    def __array_finalize__(self, obj):
        super(Array, self).__array_finalize__(obj)
        for attr in self._metadata_slots:
            setattr(self, attr, getattr(obj, attr, None))

    def __array_prepare__(self, obj, context=None):
        return super(Array, self).__array_prepare__(obj, context=context)

    def __array_wrap__(self, obj, context=None):
        return super(Array, self).__array_wrap__(obj, context=context)

    def copy(self, order='C'):
        new = super(Array, self).copy(order=order)
        new.__dict__ = self.copy_metadata()
        return new
    copy.__doc__ = Quantity.copy.__doc__

    def copy_metadata(self):
        """Return a deepcopy of the metadata for this array
        """
        return deepcopy(self.__dict__)

    def __repr__(self):
        """Return a representation of this object

        This just represents each of the metadata objects appropriately
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
    # Pickle helpers

    def dumps(self, order='C'):
        return super(Quantity, self).dumps()
    dumps.__doc__ = numpy.ndarray.dumps.__doc__

    def tostring(self, order='C'):
        return super(Quantity, self).tostring()
    tostring.__doc__ = numpy.ndarray.tostring.__doc__

    # -------------------------------------------
    # array methods

    def median(self, axis=None, **kwargs):
        return self._wrap_function(numpy.median, axis, **kwargs)
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
        """Name for this data set

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
        """GPS epoch associated with these data

        :type: `~astropy.time.Time`
        """
        if self._epoch is None:
            return None
        else:
            return Time(float(to_gps(self._epoch)),
                        format='gps', scale='utc')

    @epoch.setter
    def epoch(self, epoch):
        if epoch is None:
            self._epoch = None
        else:
            self._epoch = to_gps(epoch)

    @property
    def channel(self):
        """Instrumental channel associated with these data

        :type: `~gwpy.detector.Channel`
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
        """The physical unit of these data

        :type: `~astropy.units.UnitBase`
        """
        try:
            return self._unit
        except AttributeError:
            return None

    @unit.setter
    def unit(self, unit):
        if not hasattr(self, '_unit') or self._unit is None:
            self._unit = parse_unit(unit)
        else:
            raise AttributeError(
                "Can't set attribute. To change the units of this %s, use the "
                ".to() instance method instead, otherwise use the "
                "override_unit() instance method to forcefully set a new unit."
                % type(self).__name__)

    @unit.deleter
    def unit(self):
        del self._unit

    # -------------------------------------------
    # unit manipulations

    def _to_own_unit(self, value, check_precision=True):
        if self.unit is None:
            try:
                self.unit = ''
                return super(Array, self)._to_own_unit(
                    value, check_precision=check_precision)
            finally:
                del self.unit
        else:
            return super(Array, self)._to_own_unit(
                value, check_precision=check_precision)
    _to_own_unit.__doc__ = Quantity._to_own_unit.__doc__

    def override_unit(self, unit, parse_strict='raise'):
        """Forcefully reset the unit of these data

        Use of this method is discouraged in favour of `to()`,
        which performs accurate conversions from one unit to another.
        The method should really only be used when the original unit of the
        array is plain wrong.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, `str`
            the unit to force onto this array

        Raises
        ------
        ValueError
            if a `str` cannot be parsed as a valid unit
        """
        self._unit = parse_unit(unit, parse_strict=parse_strict)

    # -------------------------------------------
    # extras

    # use input/output registry to allow multi-format reading
    read = classmethod(reader())
    write = writer()

    def to_hdf5(self, *args, **kwargs):
        """Convert this array to a :class:`h5py.Dataset`.

        This method has been deprecated in favour of the unified I/O method:

        Class.write(..., format='hdf5')
        """
        warnings.warn("The {0}.to_hdf5 and {0}.from_hdf5 methods have been "
                      "deprecated and will be removed in an upcoming release. "
                      "Please use the unified I/O methods {0}.read() and "
                      "{0}.write() with the `format='hdf5'` keyword "
                      "argument".format(type(self).__name__),
                      DeprecationWarning)
        kwargs.setdefault('format', 'hdf5')
        return self.write(*args, **kwargs)

    @classmethod
    def from_hdf5(cls, *args, **kwargs):
        """Read an array from the given HDF file.

        This method has been deprecated in favour of the unified I/O method:

        Class.write(..., format='hdf5')
        """
        warnings.warn("The {0}.to_hdf5 and {0}.from_hdf5 methods have been "
                      "deprecated and will be removed in an upcoming release. "
                      "Please use the unified I/O methods {0}.read() and "
                      "{0}.write() with the `format='hdf5'` keyword "
                      "argument".format(cls.__name__), DeprecationWarning)
        kwargs.setdefault('format', 'hdf5')
        return cls.read(*args, **kwargs)
