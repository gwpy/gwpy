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
import inspect
from scipy import signal

from astropy.units import (Unit, Quantity)

from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

from .array import Array
from ..segments import Segment


class Series(Array):
    """A one-dimensional array with metadata
    """
    _metadata_slots = Array._metadata_slots + ['x0', 'dx', 'xunit', 'logx']
    xunit = Unit('')
    def __new__(cls, data, dtype=None, copy=False, subok=True, **metadata):
        """Define a new `Series`
        """
        if isinstance(data, (list, tuple)):
           data = numpy.asarray(data)
        if not data.ndim == 1:
            raise ValueError("Cannot create a %s with more than one "
                             "dimension" % cls.__name__)
        return super(Series, cls).__new__(cls, data, dtype=dtype, copy=copy,
                                          subok=subok, **metadata)

    # simple slice, need to copy x0 away from self
    def __getslice__(self, i, j):
        new = super(Series, self).__getslice__(i, j)
        new = new.copy()
        new.x0 = float(self.x0.value)
        new.x0 += (i * new.dx)
        return new

    # rebuild getitem to handle complex slicing
    def __getitem__(self, item):
        if isinstance(item, (float, int)):
            return Quantity(super(Series, self).__getitem__(item), self.unit)
        elif isinstance(item, slice):
            item = slice(item.start is not None and int(item.start) or None,
                         item.stop is not None and int(item.stop) or None,
                         item.step is not None and int(item.step) or None)
            new = super(Series, self).__getitem__(item)
            if item.start:
                new.x0 = float(self.x0.value)
                new.x0 += item.start * new.dx
            if item.step:
                new.dx = float(self.dx.value)
                new.dx *= item.step
            return new
        else:
            new = super(Series, self).__getitem__(item)
            if isinstance(item, (list, tuple, numpy.ndarray)):
                new.index = self.index[item]
            else:
                new.index = self.index[item.argmax()]
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
    def span(self):
        """Extent of this `Series`
        """
        return Segment(self.x0.value,
                       self.x0.value + self.shape[0] * self.dx.value)

    @property
    def index(self):
        """Positions of the data on the x-axis

        :type: `Series`
        """
        try:
            return self._index
        except AttributeError:
            if not self.size:
                self.index = numpy.ndarray(0)
            elif self.logx:
                logdx = (numpy.log10(self.x0.value + self.dx.value) -
                         numpy.log10(self.x0.value))
                logx1 = numpy.log10(self.x0.value) + self.shape[-1] * logdx
                self.index = numpy.logspace(numpy.log10(self.x0.value), logx1,
                                             num=self.shape[-1])
            else:
                self.index = (numpy.arange(self.shape[-1]) * self.dx.value +
                              self.x0.value)
            return self.index

    @index.setter
    def index(self, samples):
        if not isinstance(samples, Array):
            fname = inspect.stack()[0][3]
            name = '%s %s' % (self.name, fname)
            samples = Array(samples, unit=self.xunit, name=name,
                            epoch=self.epoch, channel=self.channel, copy=True)
        self._index = samples
        try:
            self.x0 = self.index[0]
        except IndexError:
            pass
        try:
            self.dx = self.index[1] - self.index[0]
        except IndexError:
            pass

    @property
    def logx(self):
        """Boolean telling whether this `Series` has a logarithmic
        x-axis scale
        """
        try:
            return self.metadata['logx']
        except KeyError:
            self.logx = False
            return self.logx

    @logx.setter
    def logx(self, val):
        if (val and 'logx' in self.metadata and not self.metadata['logx'] and
                'index' in self.metadata):
            del self.index
        self.metadata['logx'] = bool(val)

    # -------------------------------------------
    # Series methods

    def resample(self, rate, window=None, dtype=None):
        """Resample this Series to a new rate

        Parameters
        ----------
        rate : `float`
            rate to which to resample this `Series`
        window : array_like, callable, string, float, or tuple, optional
            specifies the window applied to the signal in the Fourier
            domain.
        dtype : :class:`numpy.dtype`, default: `None`
            specific data type for output, defaults to input dtype

        Returns
        -------
        Series
            a new Series with the resampling applied, and the same
            metadata
        """
        if isinstance(rate, Quantity):
            rate = rate.value
        N = int(self.shape[0] * self.dx.value * rate)
        data = signal.resample(self.data, N, window=window)
        new = self.__class__(data, dtype=dtype or self.dtype)
        new.metadata = self.metadata.copy()
        new.dx = 1 / float(rate)
        return new

    def decimate(self, q, n=None, ftype='iir', axis=-1):
        """Downsample the signal by using a filter.

        By default, an order 8 Chebyshev type I filter is used.
        A 30 point FIR filter with hamming window is used if
        `ftype` is 'fir'.

        Parameters
        ----------
        q : `int`
            the downsampling factor.
        n : `int`, optional
            the order of the filter (1 less than the length for 'fir').
        ftype : `str`: {'iir', 'fir'}, optional
            the type of the lowpass filter.
        axis : `int`, optional
            The axis along which to decimate.

        Returns
        -------
        y : ndarray
        The down-sampled signal.

        See also
        --------
        :meth:`scipy.resample`
        """
        if not isinstance(q, int):
            raise TypeError("q must be an integer")
        if n is None:
            if ftype == 'fir':
                n = 30
            else:
                n = 8
        if ftype == 'fir':
            b = signal.firwin(n + 1, 1. / q, window='hamming')
            a = 1.
        else:
            b, a = signal.cheby1(n, 0.05, 0.8 / q)
        y = signal.lfilter(b, a, self.data, axis=axis)
        out = self.__class__(y, **self.metadata)
        sl = [slice(None)] * y.ndim
        sl[axis] = slice(None, None, q)
        return out[sl]

    # -------------------------------------------
    # numpy.ndarray method modifiers
    # all of these try to return Quantities rather than simple numbers

    def max(self, *args, **kwargs):
        return Quantity(super(Series, self).max(*args, **kwargs),
                        unit=self.unit)
    max.__doc__ = Array.max.__doc__

    def min(self, *args, **kwargs):
        return Quantity(super(Series, self).min(*args, **kwargs),
                        unit=self.unit)
    min.__doc__ = Array.min.__doc__

    def mean(self, *args, **kwargs):
        return Quantity(super(Series, self).mean(*args, **kwargs),
                        unit=self.unit)
    mean.__doc__ = Array.mean.__doc__

    def median(self, *args, **kwargs):
        return Quantity(super(Series, self).median(*args, **kwargs),
                        unit=self.unit)
    median.__doc__ = Array.median.__doc__
