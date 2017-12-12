# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013-2016)
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

from warnings import warn

import numpy

from astropy.units import (Unit, Quantity)

from .series import Series
from .index import Index

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class Array2D(Series):
    """A two-dimensional array with metadata

    Parameters
    ----------
    value : array-like
        input data array

    unit : `~astropy.units.Unit`, optional
        physical unit of these data

    x0 : `float`, `~astropy.units.Quantity`, optional, default: `0`
        the starting value for the x-axis of this array

    dx : `float`, `~astropy.units.Quantity, optional, default: `1`
        the step size for the x-axis of this array

    xindex : `array-like`
        the complete array of x-axis values for this array. This argument
        takes precedence over `x0` and `dx` so should be
        given in place of these if relevant, not alongside

    xunit : `~astropy.units.Unit`, optional
        the unit of the x-axis coordinates. If not given explicitly, it will be
        taken from any of `dx`, `x0`, or `xindex`, or set to a boring default

    y0 : `float`, `~astropy.units.Quantity`, optional, default: `0`
        the starting value for the y-axis of this array

    dy : `float`, `~astropy.units.Quantity, optional, default: `1`
        the step size for the y-axis of this array

    yindex : `array-like`
        the complete array of y-axis values for this array. This argument
        takes precedence over `y0` and `dy` so should be
        given in place of these if relevant, not alongside

    yunit : `~astropy.units.Unit`, optional
        the unit of the y-axis coordinates. If not given explicitly, it will be
        taken from any of `dy`, `y0`, or `yindex`, or set to a boring default

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine

    name : `str`, optional
        descriptive title for this array

    channel : `~gwpy.detector.Channel`, `str`, optional
        source data stream for these data

    dtype : `~numpy.dtype`, optional
        input data type

    copy : `bool`, optional, default: `False`
        choose to copy the input data to new memory

    subok : `bool`, optional, default: `True`
        allow passing of sub-classes by the array generator

    Returns
    -------
    array : `Array`
        a new array, with a view of the data, and all associated metadata
    """
    _metadata_slots = Series._metadata_slots + ('y0', 'dy', 'yindex')
    _default_xunit = Unit('')
    _default_yunit = Unit('')
    _rowclass = Series
    _columnclass = Series
    _ndim = 2

    def __new__(cls, data, unit=None,
                x0=None, dx=None, xindex=None, xunit=None,
                y0=None, dy=None, yindex=None, yunit=None, **kwargs):
        """Define a new `Array2D`
        """

        # create new object
        new = super(Array2D, cls).__new__(cls, data, unit=unit, xindex=xindex,
                                          x0=x0, dx=dx, **kwargs)

        # set y-axis metadata from yindex
        if yindex is not None:
            # warn about duplicate settings
            if dy is not None:
                warn("yindex was given to %s(), dy will be ignored"
                     % cls.__name__)
            if y0 is not None:
                warn("yindex was given to %s(), y0 will be ignored"
                     % cls.__name__)
            # get unit
            if yunit is None and isinstance(yindex, Quantity):
                yunit = yindex.unit
            elif yunit is None:
                yunit = cls._default_yunit
            new.yindex = Quantity(yindex, unit=yunit)
        # or from y0 and dy
        else:
            if yunit is None and isinstance(dy, Quantity):
                yunit = dy.unit
            elif yunit is None and isinstance(y0, Quantity):
                yunit = y0.unit
            elif yunit is None:
                yunit = cls._default_yunit
            if dy is not None:
                new.dy = Quantity(dy, yunit)
            if y0 is not None:
                new.y0 = Quantity(y0, yunit)

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
            if isinstance(x, (float, int)):
                new = new.view(self._columnclass)
                new.dx = self.dy
                new.x0 = self.y0
                new.__metadata_finalize__(self)
                try:
                    new.xindex = self._yindex
                except AttributeError:
                    pass
            else:
                new = new.view(self._rowclass)
                new.__metadata_finalize__(self)
        # unwrap a Spectrogram
        else:
            new = new.view(type(self))
        # update metadata (Series.__getitem__ has already done x slice)
        if len(new.shape) == 1 and isinstance(y, slice):  # FrequencySeries
            try:
                self._xindex
            except AttributeError:
                if y.start:
                    new.x0 = self.x0 + y.start * self.dx
                if y.step:
                    new.dx = self.dx * y.step
            else:
                new.xindex = self.xindex[y]
        elif isinstance(y, slice):  # slice Array2D y-axis
            try:
                self._yindex
            except AttributeError:
                if y.start:
                    new.y0 = new.y0 + y.start * new.dy
                if y.step:
                    new.dy = new.dy * y.step
            else:
                new.yindex = self.yindex[y]
        return new

    def __array_finalize__(self, obj):
        super(Array2D, self).__array_finalize__(obj)
        # Series.__array_finalize__ might set _yindex to None, so delete it
        if getattr(self, '_yindex', 0) is None:
            del self.yindex

    def __iter__(self):
        # astropy Quantity.__iter__ does something fancy that we don't need
        # because we overload __getitem__
        return super(Quantity, self).__iter__()

    # -- Array2d properties ---------------------

    # y0
    @property
    def y0(self):
        """Y-axis coordinate of the first data point

        :type: `~astropy.units.Quantity` scalar
        """
        try:
            return self._y0
        except AttributeError:
            self._y0 = Quantity(0, self.yunit)
            return self._y0

    @y0.setter
    def y0(self, value):
        if value is None:
            del self.y0
            return
        if not isinstance(value, Quantity):
            try:
                value = Quantity(value, self.yunit)
            except TypeError:
                value = Quantity(float(value), self.yunit)
        # if setting new y0, delete yindex
        try:
            y0 = self._y0
        except AttributeError:
            del self.yindex
        else:
            if value is None or self.y0 is None or value != y0:
                del self.yindex
        self._y0 = value

    @y0.deleter
    def y0(self):
        try:
            del self._y0
        except AttributeError:
            pass

    # dy
    @property
    def dy(self):
        """Y-axis sample separation

        :type: `~astropy.units.Quantity` scalar
        """
        try:
            return self._dy
        except AttributeError:
            try:
                self._yindex
            except AttributeError:
                self._dy = Quantity(1, self.yunit)
            else:
                if not self.yindex.regular:
                    raise AttributeError(
                        "This series has an irregular y-axis "
                        "index, so 'dy' is not well defined")
                self._dy = self.yindex[1] - self.yindex[0]
            return self._dy

    @dy.setter
    def dy(self, value):
        # delete if None
        if value is None:
            del self.dy
            return
        # convert float to Quantity
        if not isinstance(value, Quantity):
            value = Quantity(value).to(self.yunit)
        # if value is changing, delete yindex
        try:
            dy = self._dy
        except AttributeError:
            del self.yindex
        else:
            if value is None or self.dy is None or value != dy:
                del self.yindex
        self._dy = value

    @dy.deleter
    def dy(self):
        try:
            del self._dy
        except AttributeError:
            pass

    @property
    def yunit(self):
        """Unit of Y-axis index

        :type: `~astropy.units.Unit`
        """
        try:
            return self._dy.unit
        except AttributeError:
            try:
                return self._y0.unit
            except AttributeError:
                return self._default_yunit

    # yindex
    @property
    def yindex(self):
        """Positions of the data on the y-axis

        :type: `~astropy.units.Quantity` array
        """
        try:
            return self._yindex
        except AttributeError:
            # create regular index on-the-fly
            self._yindex = Index(
                self.y0 + (numpy.arange(self.shape[1]) * self.dy), copy=False)
            return self._yindex

    @yindex.setter
    def yindex(self, index):
        if index is None:
            del self.yindex
            return
        if not isinstance(index, Index):
            try:
                unit = index.unit
            except AttributeError:
                unit = self._default_yunit
            index = Index(index, unit=unit, copy=False)
        self.y0 = index[0]
        if index.regular:
            self.dy = index[1] - index[0]
        else:
            del self.dy
        self._yindex = index

    @yindex.deleter
    def yindex(self):
        try:
            del self._yindex
        except AttributeError:
            pass

    @property
    def yspan(self):
        """Y-axis [low, high) segment encompassed by these data

        :type: `~gwpy.segments.Segment`
        """
        from ..segments import Segment
        try:
            dy = self.dy.to(self.yunit).value
        except AttributeError:  # irregular yindex
            try:
                dy = self.yindex.value[-1] - self.yindex.value[-2]
            except IndexError:
                raise ValueError("Cannot determine y-axis stride (dy)"
                                 "from a single data point")
            return Segment(self.yindex.value[0], self.yindex.value[-1] + dy)
        else:
            y0 = self.y0.to(self.yunit).value
            return Segment(y0, y0+self.shape[1]*dy)

    # -- Array2D methods ------------------------

    def is_compatible(self, other):
        """Check whether this array and ``other`` have compatible metadata
        """
        super(Array2D, self).is_compatible(other)
        # check y-axis metadata
        if isinstance(other, type(self)):
            try:
                if not self.dy == other.dy:
                    raise ValueError("%s sample sizes do not match: "
                                     "%s vs %s." % (type(self).__name__,
                                                    self.dy, other.dy))
            except AttributeError:
                raise ValueError("Series with irregular y-indexes cannot "
                                 "be compatible")

        return True

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
        except IndexError as exc:
            exc.args = ("Value %r not found in array xindex" % x,)
            raise
        try:
            idy = (self.yindex.value == y).nonzero()[0][0]
        except IndexError as exc:
            exc.args = ("Value %r not found in array yindex",)
            raise
        return self[idx, idy]

    # -- Array2D modifiers ----------------------
    # all of these try to return Quantities rather than simple numbers

    def _wrap_function(self, function, *args, **kwargs):
        out = super(Array2D, self)._wrap_function(function, *args, **kwargs)
        if out.ndim == 1:  # return Series
            # HACK: need to check astropy will always pass axis as first arg
            axis = args[0]
            metadata = {'unit': out.unit, 'channel': out.channel,
                        'epoch': self.epoch,
                        'name': '%s %s' % (self.name, function.__name__)}
            # return Column series
            if axis == 0:
                if hasattr(self, '_yindex'):
                    metadata['xindex'] = self.yindex
                else:
                    metadata['x0'] = self.y0
                    metadata['dx'] = self.dy
                return self._columnclass(out.value, **metadata)
            # return Row series
            if hasattr(self, '_xindex'):
                metadata['xindex'] = self.xindex
            else:
                metadata['x0'] = self.x0
                metadata['dx'] = self.dx
            return self._rowclass(out.value, **metadata)
        return out
