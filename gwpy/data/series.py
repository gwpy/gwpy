# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""The `Series` is a one-dimensional array with metadata
"""

import numpy
import inspect
from scipy import signal

from astropy.units import (Unit, Quantity)

from ..version import version as __version__
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

    # rebuild getitem to handle complex slicing
    def __getitem__(self, item):
        new = super(Series, self).__getitem__(item)
        if isinstance(item, int):
            return Quantity(new, unit=self.unit)
        elif isinstance(item, slice):
            if item.start:
                x0 = self.x0.copy()
                new.x0 += (item.start * new.dx)
                self.x0 = x0
            if item.step:
                new.dx *= item.step
        elif isinstance(item, (list, tuple)):
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
        return Segment(self.x0, self.x0 + self.shape[0] * self.dx)

    @property
    def index(self):
        """Positions of the data on the x-axis

        :type: `Series`
        """
        try:
            return self._index
        except AttributeError:
            if self.logx:
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
                            epoch=self.epoch, channel=self.channel)
        self._index = samples
        self.x0 = self.index[0]
        try:
            self.dx = self.index[1] - self.index[0]
        except IndexError:
            del self.dx

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
        if (val and self.metadata.has_key('logx') and
            not self.metadata['logx'] and self.metadata.has_key('index')):
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

        Returns
        -------
        Series
            a new Series with the resampling applied, and the same
            metadata
        """
        if isinstance(rate, Quantity):
            rate = rate.value
        N = self.shape[0] * self.dx * rate
        data = signal.resample(self.data, N, window=window)
        new = self.__class__(data, dtype=dtype or self.dtype)
        new.metadata = self.metadata.copy()
        new.dx = 1 / float(rate)
        return new

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
        return super(Series, self).min(*args, **kwargs) * self.unit
    min.__doc__ = Array.min.__doc__

    def mean(self, *args, **kwargs):
        return Quantity(super(Series, self).mean(*args, **kwargs),
                        unit=self.unit)
        return super(Series, self).mean(*args, **kwargs) * self.unit
    mean.__doc__ = Array.mean.__doc__

    def median(self, *args, **kwargs):
        return Quantity(super(Series, self).median(*args, **kwargs),
                        unit=self.unit)
    median.__doc__ = Array.median.__doc__
