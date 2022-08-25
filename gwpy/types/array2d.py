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

"""The `Series` is a one-dimensional array with metadata
"""

from warnings import warn


from astropy.units import (Unit, Quantity)

from . import sliceutils
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
        new = super().__new__(cls, data, unit=unit, xindex=xindex,
                              xunit=xunit, x0=x0, dx=dx, **kwargs)

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
        new = super().__getitem__(item)

        # slice axis 1 metadata
        colslice, rowslice = sliceutils.format_nd_slice(item, self.ndim)

        # column slice
        if new.ndim == 1 and isinstance(colslice, int):
            new = new.view(self._columnclass)
            del new.xindex
            new.__metadata_finalize__(self)
            sliceutils.slice_axis_attributes(self, 'y', new, 'x', rowslice)

        # row slice
        elif new.ndim == 1:
            new = new.view(self._rowclass)

        # slice axis 1 for Array2D (Series.__getitem__ will have performed
        #                           column slice already)
        elif new.ndim > 1 and not sliceutils.null_slice(rowslice):
            sliceutils.slice_axis_attributes(self, 'y', new, 'y', rowslice)

        return new

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
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
            try:
                self._y0 = self._yindex[0]
            except (AttributeError, IndexError):
                self._y0 = Quantity(0, self.yunit)
            return self._y0

    @y0.setter
    def y0(self, value):
        self._update_index("y", "y0", value)

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
        self._update_index("y", "dy", value)

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
            self._yindex = Index.define(self.y0, self.dy, self.shape[1])
            return self._yindex

    @yindex.setter
    def yindex(self, index):
        self._set_index("yindex", index)

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
        return self._index_span("y")

    @property
    def T(self):
        trans = self.value.T.view(type(self))
        trans.__array_finalize__(self)
        if hasattr(self, '_xindex'):
            trans.yindex = self.xindex.view()
        else:
            trans.y0 = self.x0
            trans.dy = self.dx
        if hasattr(self, '_yindex'):
            trans.xindex = self.yindex.view()
        else:
            trans.x0 = self.y0
            trans.dx = self.dy
        return trans

    # -- Array2D methods ------------------------

    def _is_compatible_gwpy(self, other):
        self._compare_index(other, "y")
        return super()._is_compatible_gwpy(other)

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

    def imshow(self, **kwargs):
        return self.plot(method='imshow', **kwargs)

    def pcolormesh(self, **kwargs):
        return self.plot(method='pcolormesh', **kwargs)

    def plot(self, method="imshow", **kwargs):
        from ..plot import Plot

        # correct for log scales and zeros
        if kwargs.get('xscale') == 'log' and self.x0.value == 0:
            kwargs.setdefault('xlim', (self.dx.value, self.xspan[1]))
        if kwargs.get('yscale') == 'log' and self.y0.value == 0:
            kwargs.setdefault('ylim', (self.dy.value, self.yspan[1]))

        # make plot
        return Plot(self, method=method, **kwargs)

    # -- Array2D modifiers ----------------------
    # all of these try to return Quantities rather than simple numbers

    def _wrap_function(self, function, *args, **kwargs):
        out = super()._wrap_function(function, *args, **kwargs)
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
