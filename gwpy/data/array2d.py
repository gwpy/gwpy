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

import math
import inspect

import numpy
from scipy import signal
from astropy.units import (Unit, Quantity)

from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

from .array import Array
from .series import Series
from ..segments import Segment


class Array2D(Array):
    """A two-dimensional array with metadata
    """
    _metadata_slots = (Array._metadata_slots +
                       ['x0', 'dx', 'y0', 'dy', 'logx', 'logy'])
    xunit = Unit('')
    yunit = Unit('')

    def __new__(cls, data, dtype=None, copy=False, subok=True, **metadata):
        """Define a new `Array2D`
        """
        if len(data) and not numpy.asarray(data).ndim == 2:
            raise ValueError("Data must be two-dimensional")
        return super(Array2D, cls).__new__(cls, data, dtype=dtype, copy=copy,
                                           subok=subok, **metadata)

    # simple slice, need to copy x0 away from self
    def __getslice__(self, i, j):
        new = super(Array2D, self).__getslice__(i, j).copy()
        new.x0 = float(self.x0.value)
        new.x0 += (i * new.dx)
        return new

    # rebuild getitem to handle complex slicing
    def __getitem__(self, item):
        new = super(Array2D, self).__getitem__(item).copy()
        # if given an int, extract a column
        if isinstance(item, int):
            new = Series(new, unit=self.unit, name=self.name, dx=self.dy,
                         epoch=self.epoch, channel=self.channel, x0=self.y0)
            new.xunit = self.yunit
            return new
        # if given a tuple, extract an element
        elif isinstance(item, tuple) and len(item) == 1:
            return Quantity(new, unit=self.unit)
        # otherwise perform complex slice
        elif isinstance(item, slice):
            if item.start:
                new.x0 += item.start * self.dx
            if item.step:
                new.dx *= item.step
            return new
        else:
            return new

    # -------------------------------------------
    # Series properties

    @property
    def x0(self):
        """X-axis value of first sample
        """
        return self.metadata['x0']

    @x0.setter
    def x0(self, value):
        if isinstance(value, Quantity):
            self.metadata['x0'] = value.to(self.xunit)
        else:
            self.metadata['x0'] = Quantity(value, self.xunit)

    @x0.deleter
    def x0(self):
        del self.metadata['x0']

    @property
    def dx(self):
        """Distance between samples on the x-axis
        """
        return self.metadata['dx']

    @dx.setter
    def dx(self, value):
        if isinstance(value, Quantity):
            self.metadata['dx'] = value.to(self.xunit)
        else:
            self.metadata['dx'] = Quantity(value, self.xunit)

    @dx.deleter
    def dx(self):
        del self.metadata['dx']

    @property
    def span_x(self):
        """Extent of this `Array2D`
        """
        return Segment(self.x0, self.x0 + self.shape[0] * self.dx)

    @property
    def y0(self):
        """X-axis value of first sample
        """
        return self.metadata['y0']

    @y0.setter
    def y0(self, value):
        if isinstance(value, Quantity):
            self.metadata['y0'] = value.to(self.yunit)
        else:
            self.metadata['y0'] = Quantity(value, self.yunit)

    @y0.deleter
    def y0(self):
        del self.metadata['y0']

    @property
    def dy(self):
        """Distance between samples on the x-axis
        """
        return self.metadata['dy']

    @dy.setter
    def dy(self, value):
        if isinstance(value, Quantity):
            self.metadata['dy'] = value.to(self.yunit)
        else:
            self.metadata['dy'] = Quantity(value, self.yunit)

    @dy.deleter
    def dy(self):
        del self.metadata['dy']

    @property
    def span_y(self):
        """Extent of this `Array2D`
        """
        return Segment(self.y0, self.y0 + self.shape[1] * self.dy)

    @property
    def xindex(self):
        """Positions of the data on the x-axis

        :type: `Series`
        """
        try:
            return self._xindex
        except AttributeError:
            if self.logx:
                logdx = (numpy.log10(self.x0.value + self.dx.value) -
                         numpy.log10(self.x0.value))
                logx1 = numpy.log10(self.x0.value) + self.shape[0] * logdx
                self.xindex = numpy.logspace(math.log10(self.x0.value), logx1,
                                             num=self.shape[0])
            else:
                self.xindex = (numpy.arange(self.shape[0]) * self.dx.value +
                               self.x0.value)
            return self.xindex

    @xindex.setter
    def xindex(self, samples):
        if not isinstance(samples, Array):
            fname = inspect.stack()[0][3]
            name = '%s %s' % (self.name, fname)
            samples = Array(samples, unit=self.xunit, name=name,
                            epoch=self.epoch, channel=self.channel)
        self._xindex = samples
        self.x0 = self.xindex[0]
        try:
            self.dx = self.xindex[1] - self.xindex[0]
        except IndexError:
            pass

    @property
    def yindex(self):
        """Positions of the data on the y-axis

        :type: `Series`
        """
        try:
            return self._yindex
        except AttributeError:
            if self.logy:
                logdy = (numpy.log10(self.y0.value + self.dy.value) -
                         numpy.log10(self.y0.value))
                logy1 = numpy.log10(self.y0.value) + self.shape[-1] * logdy
                self.yindex = numpy.logspace(math.log10(self.y0.value), logy1,
                                             num=self.shape[-1])
            else:
                self.yindex = (numpy.arange(self.shape[-1]) * self.dy.value +
                               self.y0.value)
            return self.yindex

    @yindex.setter
    def yindex(self, samples):
        if not isinstance(samples, Array):
            fname = inspect.stack()[0][3]
            name = '%s %s' % (self.name, fname)
            samples = Array(samples, unit=self.yunit, name=name,
                            epoch=self.epoch, channel=self.channel)
        self._yindex = samples
        self.y0 = self.yindex[0]
        try:
            self.dy = self.yindex[1] - self.yindex[0]
        except IndexError:
            pass

    @property
    def logx(self):
        """Boolean telling whether this `Array2D` has a logarithmic
        x-axis scale
        """
        try:
            return self.metadata['logx']
        except KeyError:
            self.logx = False
            return self.logx

    @logx.setter
    def logx(self, val):
        if (val and 'logx' in self.metadata and not self.metdata['logx'] and
                hasattr(self, '_xindex')):
            del self._xindex
        self.metadata['logx'] = bool(val)

    @property
    def logy(self):
        """Boolean telling whether this `Array2D` has a logarithmic
        y-ayis scale
        """
        try:
            return self.metadata['logy']
        except KeyError:
            self.logy = False
            return self.logy

    @logy.setter
    def logy(self, val):
        if (val and 'logy' in self.metadata and not self.metadata['logy'] and
                hasattr(self, '_yindex')):
            del self._index
        self.metadata['logy'] = bool(val)

    # -------------------------------------------
    # Array2D methods

    def resample(self, rate, window=None):
        """Resample this Array2D to a new rate

        Parameters
        ----------
        rate : `float`
            rate to which to resample this `Array2D`
        window : array_like, callable, string, float, or tuple, optional
            specifies the window applied to the signal in the Fourier
            domain.

        Returns
        -------
        Array2D
            a new Array2D with the resampling applied, and the same
            metadata
        """
        if isinstance(rate, Quantity):
            rate = rate.value
        n = self.size * self.dx * rate
        data = signal.resample(self.data, n, window=window)
        new = self.__class__(data, **self.metadata)
        new.dx = 1 / rate
        return new

    # -------------------------------------------
    # numpy.ndarray method modifiers
    # all of these try to return Quantities rather than simple numbers

    def max(self, *args, **kwargs):
        out = super(Array2D, self).max(*args, **kwargs)
        if isinstance(out, Array) and out.shape:
            return Series(out, name='%s max' % self.name, unit=self.unit,
                          x0=out.y0.value, dx=out.dy.value)
        else:
            return out * self.unit
    max.__doc__ = Array.max.__doc__

    def min(self, *args, **kwargs):
        out = super(Array2D, self).min(*args, **kwargs)
        if isinstance(out, Array) and out.shape:
            return Series(out, name='%s min' % self.name, unit=self.unit,
                          x0=out.y0.value, dx=out.dy.value)
        else:
            return out * self.unit
    min.__doc__ = Array.min.__doc__

    def mean(self, *args, **kwargs):
        out = super(Array2D, self).mean(*args, **kwargs)
        if isinstance(out, Array) and out.shape:
            return Series(out, name='%s mean' % self.name, unit=self.unit,
                          x0=out.y0.value, dx=out.dy.value)
        else:
            return out * self.unit
    mean.__doc__ = Array.mean.__doc__

    def median(self, *args, **kwargs):
        out = super(Array2D, self).median(*args, **kwargs)
        if isinstance(out, Array) and out.ndim == 1:
            return Series(out.data, name='%s median' % self.name,
                          unit=self.unit, x0=self.y0.value, dx=self.dy.value)
        else:
            return out * self.unit
    median.__doc__ = Array.median.__doc__

    def __array_wrap__(self, obj, context=None):
        """Wrap an array as an `Array2D` with metadata
        """
        result = obj.view(self.__class__)
        result.metadata = self.metadata.copy()
        try:
            result._xindex = self._xindex
        except AttributeError:
            pass
        try:
            result._yindex = self._yindex
        except AttributeError:
            pass
        return result
