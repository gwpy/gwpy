# coding=utf-8
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

from .series import Series
from ..utils.docstring import interpolate_docstring

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

interpolate_docstring.update(
    ArrayYaxis=(
        """y0 : `float`, `~astropy.units.Quantity`, optional, default: `0`
        the starting value for the y-axis of this array

    dy : `float`, `~astropy.units.Quantity, optional, default: `1`
        the step size for the y-axis of this array

    yindex : `array-like`
        the complete array of y-axis values for this array. This argument
        takes precedence over `y0` and `dy` so should be
        given in place of these if relevant, not alongside"""),
)


@interpolate_docstring
class Array2D(Series):
    """A two-dimensional array with metadata

    Parameters
    ----------
    %(Array1)s

    %(ArrayXaxis)s

    %(ArrayYaxis)s

    %(Array2)s

    Returns
    -------
    array : `Array`
        a new array, with a view of the data, and all associated metadata
    """
    _metadata_slots = Series._metadata_slots + ['y0', 'dy', 'yindex']
    _default_xunit = Unit('')
    _default_yunit = Unit('')
    _ndim = 2

    def __new__(cls, data, unit=None, xindex=None, yindex=None, x0=0,
                dx=1, y0=0, dy=1, **kwargs):
        """Define a new `Array2D`
        """
        new = super(Array2D, cls).__new__(cls, data, unit=unit, xindex=xindex,
                                          x0=0, dx=dx, **kwargs)
        if isinstance(y0, Quantity):
            yunit = y0.unit
        elif isinstance(dy, Quantity):
            yunit = dy.unit
        else:
            yunit = cls._default_yunit
        if y0 is not None:
            new.y0 = Quantity(y0, yunit)
        if dy is not None:
            new.dy = Quantity(dy, yunit)
        if yindex is not None:
            new.yindex = yindex
        return new

    # rebuild getitem to handle complex slicing
    def __getitem__(self, item):
        new = super(Array2D, self).__getitem__(item)
        # unwrap item request
        if isinstance(item, tuple):
            x, y = item
        else:
            x = item
            y = None
        # extract a Quantity
        if numpy.shape(new) == ():
            return Quantity(new, unit=self.unit)
        # unwrap a Series
        if len(new.shape) == 1:
            new = new.view(Series)
            if isinstance(x, (float, int)):
                new.dx = self.dy
                new.x0 = self.y0
        # unwrap a Spectrogram
        else:
            new = new.value.view(type(self))
            new.__dict__ = self.copy_metadata()
        # update metadata
        if isinstance(x, slice):
            if x.start:
                new.x0 = new.x0 + x.start * new.dx
            if x.step:
                new.dx = new.dx + x.step
        if len(new.shape) == 1 and isinstance(y, slice):
            if y.start:
                new.x0 = new.x0 + y.start * new.dx
            if y.step:
                new.dx = new.dx * y.step
        elif isinstance(y, slice):
            if y.start:
                new.y0 = new.y0 + y.start * new.dy
            if y.step:
                new.dy = new.dy * y.step
        return new

    # -------------------------------------------
    # Array2D properties

    @property
    def yindex(self):
        """Positions of the data on the y-axis

        :type: `~astropy.units.Quantity` array
        """
        try:
            return self._yindex
        except AttributeError:
            self._yindex = self.y0 + (
                numpy.arange(self.shape[1]) * self.dy)
            return self._yindex

    @yindex.setter
    def yindex(self, index):
        if isinstance(index, Quantity):
            self._yindex = index
        elif index is None:
            del self.yindex
            return
        else:
            index = Quantity(index, self._default_yunit)
            self._yindex = index
        self.y0 = index[0]
        if index.size:
            self.dy = index[1] - index[0]
        else:
            self.dy = None

    @yindex.deleter
    def yindex(self):
        try:
            del self._yindex
        except AttributeError:
            pass

    @property
    def y0(self):
        """Y-axis value of the first data point

        :type: `~astropy.units.Quantity` scalar
        """
        return self._y0

    @y0.setter
    def y0(self, value):
        if not isinstance(value, Quantity) and value is not None:
            value = Quantity(value, self._default_yunit)
        try:
            y0 = self.y0
        except AttributeError:
            del self.yindex
        else:
            if value is None or self.y0 is None or value != y0:
                del self.yindex
        self._y0 = value

    @y0.deleter
    def y0(self):
        self._y0 = None

    @property
    def dy(self):
        return self._dy

    @dy.setter
    def dy(self, value):
        """Y-axis sample separation

        :type: `~astropy.units.Quantity` scalar
        """
        if not isinstance(value, Quantity) and value is not None:
            value = Quantity(value).to(self.yunit)
        try:
            dy = self.dy
        except AttributeError:
            del self.yindex
        else:
            if value is None or self.dy is None or value != dy:
                del self.yindex
        self._dy = value

    @dy.deleter
    def dy(self):
        self._dy = None

    @property
    def yunit(self):
        """Unit of y-axis index

        :type: `~astropy.units.Unit`
        """
        return self.y0.unit

    @property
    def yspan(self):
        """Y-axis [low, high) segment encompassed by these data

        :type: `~gwpy.segments.Segment`
        """
        from ..segments import Segment
        try:
            self._yindex
        except AttributeError:
            y0 = self.y0.to(self._default_yunit).value
            dy = self.dy.to(self._default_yunit).value
            return Segment(y0, y0+self.shape[0]*dy)
        else:
            return Segment(self.yindex.value[0],
                           self.yindex.value[-1] + self.dy.value)

    # -- Array2D methods ------------------------

    def value_at(self, x, y):
        """Return the value of this `Series` at the given `(x, y)` coordinates

        Parameters
        ----------
        x : `float`, `~astropy.units.Quantity`
            the `xindex` value at which to search
        x : `float`, `~astropy.units.Quantity`
            the `yindex` value at which to search

        Returns
        -------
        z : `~astropy.units.Quantity`
            the value of this Series at the given coordinates
        """
        x = Quantity(x, self.xindex.unit).value
        y = Quantity(y, self.yindex.unit).value
        try:
            idx = (self.xindex.value == x).nonzero()[0][0]
        except IndexError as e:
            e.args = ("Value %r not found in array xindex",)
            raise
        try:
            idy = (self.yindex.value == y).nonzero()[0][0]
        except IndexError as e:
            e.args = ("Value %r not found in array yindex",)
            raise
        print(idx, idy)
        return self[idx, idy]

    # -------------------------------------------
    # numpy.ndarray method modifiers
    # all of these try to return Quantities rather than simple numbers

    def _wrap_function(self, function, *args, **kwargs):
        out = super(Array2D, self)._wrap_function(function, *args, **kwargs)
        if out.ndim == 1:
            # HACK: need to check astropy will always pass axis as first arg
            axis = args[0]
            # return Series
            if axis == 0:
                x0 = self.y0
                dx = self.dy
                xindex = hasattr(self, '_yindex') and self.yindex or None
            else:
                x0 = self.x0
                dx = self.dx
                xindex = hasattr(self, '_xindex') and self.xindex or None
            return Series(out.value, unit=out.unit, x0=x0, dx=dx,
                          channel=out.channel, epoch=self.epoch, xindex=xindex,
                          name='%s %s' % (self.name, function.__name__))
        return out

    def __array_wrap__(self, obj, context=None):
        result = super(Array2D, self).__array_wrap__(obj, context=context)
        try:
            result._xindex = self._xindex
        except AttributeError:
            pass
        try:
            result._yindex = self._yindex
        except AttributeError:
            pass
        return result

    def copy(self, order='C'):
        new = super(Array2D, self).copy(order=order)
        try:
            new._xindex = self._xindex.copy()
        except AttributeError:
            pass
        try:
            new._yindex = self._yindex.copy()
        except AttributeError:
            pass
        return new
    copy.__doc__ = Series.copy.__doc__
