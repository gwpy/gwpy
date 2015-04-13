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

"""The `Series` is a one-dimensional array with metadata
"""

import numpy

from astropy.units import (Unit, Quantity)

from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

from .array import Array


class Series(Array):
    _metadata_slots = Array._metadata_slots + ['x0', 'dx']
    _default_xunit = Unit('')
    _ndim = 1

    def __new__(cls, value, unit=None, xindex=None, x0=0, dx=1, **kwargs):
        shape = numpy.shape(value)
        if len(shape) != cls._ndim:
            raise ValueError("Cannot generate Series with %d-dimensional data"
                             % len(shape))
        new = super(Series, cls).__new__(cls, value, unit=unit, **kwargs)
        if isinstance(x0, Quantity):
            xunit = x0.unit
        elif isinstance(dx, Quantity):
            xunit = dx.unit
        else:
            xunit = cls._default_xunit
        new.x0 = Quantity(x0, xunit)
        new.dx = Quantity(dx, xunit)
        new.xindex = xindex
        return new

    @property
    def xindex(self):
        """Positions of the data on the x-axis

        :type: `Series`
        """
        try:
            return self._xindex
        except AttributeError:
            self._xindex = self.x0 + (
                numpy.arange(self.shape[0], dtype=self.dtype) * self.dx)
            return self._xindex

    @xindex.setter
    def xindex(self, index):
        if isinstance(index, Quantity):
            self._xindex = index
        elif index is None:
            del self.xindex
            return
        else:
            index = Quantity(index, unit=self._default_xunit)
            self._xindex = index
        self.x0 = index[0]
        if index.size:
            self.dx = index[1] - index[0]
        else:
            self.dx = None

    @xindex.deleter
    def xindex(self):
        try:
            del self._xindex
        except AttributeError:
            pass

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value):
        if not isinstance(value, Quantity) and value is not None:
            value = Quantity(value, self._default_xunit)
        try:
            x0 = self.x0
        except AttributeError:
            del self.xindex
        else:
            if value is None or self.x0 is None or value != x0:
                del self.xindex
        self._x0 = value

    @x0.deleter
    def x0(self):
        self._x0 = None

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, value):
        if not isinstance(value, Quantity) and value is not None:
            value = Quantity(value).to(self.xunit)
        try:
            dx = self.dx
        except AttributeError:
            del self.xindex
        else:
            if value is None or self.dx is None or value != dx:
                del self.xindex
        self._dx = value

    @dx.deleter
    def dx(self):
        self._dx = None

    @property
    def xunit(self):
        return self.x0.unit

    def copy(self, order='C'):
        new = super(Series, self).copy(order=order)
        try:
            new._xindex = self._xindex.copy()
        except AttributeError:
            pass
        return new

    def zip(self):
        """Zip the `xindex` and `value` arrays of this `Series`

        Returns
        -------
        stacked : 2-d `numpy.ndarray`
            The array formed by stacking the the `xindex` and `value` of this
            `Series`.

        Examples
        --------
        >>> a = Series([0, 2, 4, 6, 8], xindex=[-5, -4, -3, -2, -1])
        >>> a.zip()
        array([[-5.,  0.],
               [-4.,  2.],
               [-3.,  4.],
               [-2.,  6.],
               [-1.,  8.]])

        """
        return numpy.column_stack((self.xindex.value, self.value))

    def __array_finalize__(self, obj):
        """Finalize a Array with metadata
        """
        super(Series, self).__array_finalize__(obj)
        if hasattr(self, '_xindex'):
            obj._xindex = self._xindex

    def __getslice__(self, i, j):
        new = super(Series, self).__getslice__(i, j)
        new.__dict__ = self.__dict__.copy()
        new.x0 = self.x0 + i * self.dx
        return new

    def __getitem__(self, item):
        if isinstance(item, (float, int)):
            return Quantity(self.value[item], unit=self.unit)
        new = super(Series, self).__getitem__(item)
        if isinstance(item, slice):
            if item.start:
                new.x0 = self.x0 + item.start * self.dx
            if item.step:
                new.dx = self.dx * item.step
        elif isinstance(item, numpy.ndarray):
            new.xindex = self.xindex[item]
        return new
