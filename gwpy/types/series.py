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

from numbers import Number
from warnings import warn
from math import floor

import numpy

from astropy.units import (Unit, Quantity, dimensionless_unscaled)
from astropy.io import registry as io_registry

from .array import Array
from .index import Index

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class Series(Array):
    """A one-dimensional data series

    A `Series` is defined as an array of data indexed upon an axis, meaning
    each sample maps to a position upon the axis. By convention the X axis
    is used to define the index, with the `~Series.x0`, `~Series.dx`, and
    `~Series.xindex` attributes allowing the positions of the data to be
    well defined.

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
        the unit of the x-axis index. If not given explicitly, it will be
        taken from any of `dx`, `x0`, or `xindex`, or set to a boring default

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
    series : `Series`
        a new `Series`

    Examples
    --------
    To define a `Series` of displacements at a given input laser power,
    for example:

    >>> data = Series([1, 2, 3, 2, 4, 3], unit='nm', x0=0, dx=2, xunit='W')
    >>> print(data)
    Series([ 1., 2., 3., 2., 4., 3.]
           unit: Unit("nm"),
           name: None,
           epoch: None,
           channel: None,
           x0: 0.0 W,
           dx: 2.0 W,
           xindex: [  0.   2.   4.   6.   8.  10.] W)
    """
    _metadata_slots = Array._metadata_slots + ('x0', 'dx', 'xindex')
    _default_xunit = Unit('')
    _ndim = 1

    def __new__(cls, value, unit=None, x0=None, dx=None, xindex=None,
                xunit=None, **kwargs):
        # check input data dimensions are OK
        shape = numpy.shape(value)
        if len(shape) != cls._ndim:
            raise ValueError("Cannot generate %s with %d-dimensional data"
                             % (cls.__name__, len(shape)))

        # create new object
        new = super(Series, cls).__new__(cls, value, unit=unit, **kwargs)

        # set x-axis metadata from xindex
        if xindex is not None:
            # warn about duplicate settings
            if dx is not None:
                warn("xindex was given to %s(), dx will be ignored"
                     % cls.__name__)
            if x0 is not None:
                warn("xindex was given to %s(), x0 will be ignored"
                     % cls.__name__)
            # get unit
            if xunit is None and isinstance(xindex, Quantity):
                xunit = xindex.unit
            elif xunit is None:
                xunit = cls._default_xunit
            new.xindex = Quantity(xindex, unit=xunit)
        # or from x0 and dx
        else:
            if xunit is None and isinstance(dx, Quantity):
                xunit = dx.unit
            elif xunit is None and isinstance(x0, Quantity):
                xunit = x0.unit
            elif xunit is None:
                xunit = cls._default_xunit
            if dx is not None:
                new.dx = Quantity(dx, xunit)
            if x0 is not None:
                new.x0 = Quantity(x0, xunit)
        return new

    # -- series creation ------------------------

    def __array_finalize__(self, obj):
        super(Series, self).__array_finalize__(obj)
        # Array.__array_finalize__ might set _xindex to None, so delete it
        if getattr(self, '_xindex', None) is None:
            del self.xindex

    # -- series properties ----------------------

    # x0
    @property
    def x0(self):
        """X-axis coordinate of the first data point

        :type: `~astropy.units.Quantity` scalar
        """
        try:
            return self._x0
        except AttributeError:
            self._x0 = Quantity(0, self.xunit)
            return self._x0

    @x0.setter
    def x0(self, value):
        if value is None:
            del self.x0
            return
        if not isinstance(value, Quantity):
            try:
                value = Quantity(value, self.xunit)
            except TypeError:
                value = Quantity(float(value), self.xunit)
        # if setting new x0, delete xindex
        try:
            x0 = self._x0
        except AttributeError:
            del self.xindex
        else:
            if value is None or self.x0 is None or value != x0:
                del self.xindex
        self._x0 = value

    @x0.deleter
    def x0(self):
        try:
            del self._x0
        except AttributeError:
            pass

    # dx
    @property
    def dx(self):
        """X-axis sample separation

        :type: `~astropy.units.Quantity` scalar
        """
        try:
            return self._dx
        except AttributeError:
            try:
                self._xindex
            except AttributeError:
                self._dx = Quantity(1, self.xunit)
            else:
                if not self.xindex.regular:
                    raise AttributeError("This series has an irregular x-axis "
                                         "index, so 'dx' is not well defined")
                self._dx = self.xindex[1] - self.xindex[0]
            return self._dx

    @dx.setter
    def dx(self, value):
        # delete if None
        if value is None:
            del self.dx
            return
        # convert float to Quantity
        if not isinstance(value, Quantity):
            value = Quantity(value, self.xunit)
        # if value is changing, delete xindex
        try:
            dx = self._dx
        except AttributeError:
            del self.xindex
        else:
            if value is None or self.dx is None or value != dx:
                del self.xindex
        self._dx = value

    @dx.deleter
    def dx(self):
        try:
            del self._dx
        except AttributeError:
            pass

    # xindex
    @property
    def xindex(self):
        """Positions of the data on the x-axis

        :type: `~astropy.units.Quantity` array
        """
        try:
            return self._xindex
        except AttributeError:
            # create regular index on-the-fly
            self._xindex = Index(
                self.x0 + (numpy.arange(self.shape[0]) * self.dx), copy=False)
            return self._xindex

    @xindex.setter
    def xindex(self, index):
        if index is None:
            del self.xindex
            return
        if not isinstance(index, Index):
            try:
                unit = index.unit
            except AttributeError:
                unit = self._default_xunit
            index = Index(index, unit=unit, copy=False)
        self.x0 = index[0]
        if index.regular:
            self.dx = index[1] - index[0]
        else:
            del self.dx
        self._xindex = index

    @xindex.deleter
    def xindex(self):
        try:
            del self._xindex
        except AttributeError:
            pass

    # xunit
    @property
    def xunit(self):
        """Unit of x-axis index

        :type: `~astropy.units.Unit`
        """
        try:
            return self._dx.unit
        except AttributeError:
            try:
                return self._x0.unit
            except AttributeError:
                return self._default_xunit

    @xunit.setter
    def xunit(self, unit):
        unit = Unit(unit)
        try:  # set the index, if present
            self.xindex = self._xindex.to(unit)
        except AttributeError:  # or just set the start and step
            self.dx = self.dx.to(unit)
            self.x0 = self.x0.to(unit)

    @property
    def xspan(self):
        """X-axis [low, high) segment encompassed by these data

        :type: `~gwpy.segments.Segment`
        """
        from ..segments import Segment
        try:
            dx = self.dx.to(self.xunit).value
        except AttributeError:  # irregular xindex
            try:
                dx = self.xindex.value[-1] - self.xindex.value[-2]
            except IndexError:
                raise ValueError("Cannot determine x-axis stride (dx)"
                                 "from a single data point")
            return Segment(self.xindex.value[0], self.xindex.value[-1] + dx)
        else:
            x0 = self.x0.to(self.xunit).value
            return Segment(x0, x0+self.shape[0]*dx)

    # -- series i/o -----------------------------

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data into a `Series`

        Arguments and keywords depend on the output format, see the
        online documentation for full details for each format, the
        parameters below are common to most formats.

        Parameters
        ----------
        source : `str`, :class:`~glue.lal.Cache`
            source of data, any of the following:

            - `str` path of single data file
            - `str` path of LAL-format cache file
            - :class:`~glue.lal.Cache` describing one or more data files,

        format : `str`, optional
            source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats

        Returns
        -------
        data : `Series`

        Notes
        -----"""
        return io_registry.read(cls, source, *args, **kwargs)

    def write(self, target, *args, **kwargs):
        """Write this `Series` to a file

        Arguments and keywords depend on the output format, see the
        online documentation for full details for each format, the
        parameters below are common to most formats.

        Parameters
        ----------
        target : `str`
            output filename

        format : `str`, optional
            output format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        Notes
        -----"""
        return io_registry.write(self, target, *args, **kwargs)

    # -- series methods -------------------------

    def value_at(self, x):
        """Return the value of this `Series` at the given `xindex` value

        Parameters
        ----------
        x : `float`, `~astropy.units.Quantity`
            the `xindex` value at which to search

        Returns
        -------
        y : `~astropy.units.Quantity`
            the value of this Series at the given `xindex` value
        """
        x = Quantity(x, self.xindex.unit).value
        try:
            idx = (self.xindex.value == x).nonzero()[0][0]
        except IndexError as e:
            e.args = ("Value %r not found in array index" % x,)
            raise
        return self[idx]

    def copy(self, order='C'):
        new = super(Series, self).copy(order=order)
        try:
            new._xindex = self._xindex.copy()
        except AttributeError:
            pass
        return new
    copy.__doc__ = Array.copy.__doc__

    def zip(self):
        """Zip the `xindex` and `value` arrays of this `Series`

        Returns
        -------
        stacked : 2-d `numpy.ndarray`
            The array formed by stacking the the `xindex` and `value` of this
            series

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

    def diff(self, n=1, axis=-1):
        """Calculate the n-th order discrete difference along given axis.

        The first order difference is given by ``out[n] = a[n+1] - a[n]`` along
        the given axis, higher order differences are calculated by using `diff`
        recursively.

        Parameters
        ----------
        n : int, optional
            The number of times values are differenced.
        axis : int, optional
            The axis along which the difference is taken, default is the
            last axis.

        Returns
        -------
        diff : `Series`
            The `n` order differences. The shape of the output is the same
            as the input, except along `axis` where the dimension is
            smaller by `n`.

        See Also
        --------
        numpy.diff
            for documentation on the underlying method
        """
        out = super(Array, self).diff(n=n, axis=axis)
        try:
            out.x0 = self.x0 + self.dx * n
        except AttributeError:  # irregular xindex
            out.x0 = self.xindex[n]
        return out

    def __getslice__(self, i, j):
        new = super(Series, self).__getslice__(i, j)
        if i:
            try:
                new.x0 = self.x0 + i * self.dx
            except AttributeError:  # irregular xindex
                new.x0 = self.xindex[i]
        return new

    def __getitem__(self, item):
        # if single value, convert to a simple Quantity
        if isinstance(item, Number):
            return Quantity(self.value[item], unit=self.unit)
        new = super(Series, self).__getitem__(item)
        # if we're slicing, update the x-axis properties
        if isinstance(item, slice):  # slice
            try:
                self._xindex
            except AttributeError:
                if item.start:
                    new.x0 = self.x0 + item.start * self.dx
                if item.step:
                    new.dx = self.dx * item.step
            else:
                new.xindex = self.xindex[item]
        elif isinstance(item, numpy.ndarray):  # index array
            new.xindex = self.xindex[item]
        return new

    def is_contiguous(self, other, tol=1/2.**18):
        """Check whether other is contiguous with self.

        Parameters
        ----------
        other : `Series`, `numpy.ndarray`
            another series of the same type to test for contiguity

        tol : `float`, optional
            the numerical tolerance of the test

        Returns
        -------
        1
            if `other` is contiguous with this series, i.e. would attach
            seamlessly onto the end
        -1
            if `other` is anti-contiguous with this seires, i.e. would attach
            seamlessly onto the start
        0
            if `other` is completely dis-contiguous with thie series

        Notes
        -----
        if a raw `numpy.ndarray` is passed as other, with no metadata, then
        the contiguity check will always pass
        """
        self.is_compatible(other)
        if isinstance(other, type(self)):
            if abs(float(self.xspan[1] - other.xspan[0])) < tol:
                return 1
            elif abs(float(other.xspan[1] - self.xspan[0])) < tol:
                return -1
            return 0
        elif type(other) in [list, tuple, numpy.ndarray]:
            return 1

    def is_compatible(self, other):
        """Check whether this series and other have compatible metadata

        This method tests that the `sample size <Series.dx>`, and the
        `~Series.unit` match.
        """
        if isinstance(other, type(self)):
            # check step size, if possible
            try:
                if not self.dx == other.dx:
                    raise ValueError("%s sample sizes do not match: "
                                     "%s vs %s." % (type(self).__name__,
                                                    self.dx, other.dx))
            except AttributeError:
                raise ValueError("Series with irregular xindexes cannot "
                                 "be compatible")
            # check units
            if not self.unit == other.unit and not (
                    self.unit in [dimensionless_unscaled, None] and
                    other.unit in [dimensionless_unscaled, None]):
                raise ValueError("%s units do not match: %s vs %s."
                                 % (type(self).__name__, str(self.unit),
                                    str(other.unit)))
        else:
            # assume an array-like object, and just check that the shape
            # and dtype match
            arr = numpy.asarray(other)
            if arr.ndim != self.ndim:
                raise ValueError("Dimensionality does not match")
            if arr.dtype != self.dtype:
                warn("Array data types do not match: %s vs %s"
                     % (self.dtype, other.dtype))
        return True

    def append(self, other, gap='raise', inplace=True, pad=0, resize=True):
        """Connect another series onto the end of the current one.

        Parameters
        ----------
        other : `Series`
            another series of the same type to connect to this one

        gap : `str`, optional, default: ``'raise'``
            action to perform if there's a gap between the other series
            and this one. One of

                - ``'raise'`` - raise an `Exception`
                    - ``'ignore'`` - remove gap and join data
                - ``'pad'`` - pad gap with zeros

        inplace : `bool`, optional, default: `True`
            perform operation in-place, modifying current `Series`,
            otherwise copy data and return new `Series`

            .. warning::

               inplace append bypasses the reference check in
               `numpy.ndarray.resize`, so be carefully to only use this
               for arrays that haven't been sharing their memory!

        pad : `float`, optional, default: ``0.0``
            value with which to pad discontiguous series

        resize : `bool`, optional, default: `True`
            resize this array to accommodate new data, otherwise shift the
            old data to the left (potentially falling off the start) and
            put the new data in at the end

        Returns
        -------
        series : `Series`
            a new series containing joined data sets
        """
        # check metadata
        self.is_compatible(other)
        # make copy if needed
        if not inplace:
            self = self.copy()
        # fill gap
        if self.is_contiguous(other) != 1:
            if gap == 'pad':
                ngap = floor(
                    (other.xspan[0] - self.xspan[1]) / self.dx.value + 0.5)
                if ngap < 1:
                    raise ValueError(
                        "Cannot append {0} that starts before this one:\n"
                        "    {0} 1 span: {1}\n    {0} 2 span: {2}".format(
                            type(self).__name__, self.xspan, other.xspan))
                gapshape = list(self.shape)
                gapshape[0] = int(ngap)
                padding = (numpy.ones(gapshape) * pad).astype(self.dtype)
                self.append(padding, inplace=True, resize=resize)
            elif gap == 'ignore':
                pass
            elif self.xspan[0] < other.xspan[0] < self.xspan[1]:
                raise ValueError(
                    "Cannot append overlapping {0}s:\n"
                    "    {0} 1 span: {1}\n    {0} 2 span: {2}".format(
                        type(self).__name__, self.xspan, other.xspan))
            else:
                raise ValueError(
                    "Cannot append discontiguous {0}\n"
                    "    {0} 1 span: {1}\n    {0} 2 span: {2}".format(
                        type(self).__name__, self.xspan, other.xspan))

        # check empty other
        if not other.size:
            return self

        # resize first
        if resize:
            N = other.shape[0]
            s = list(self.shape)
            s[0] = self.shape[0] + other.shape[0]
            try:
                self.resize(s, refcheck=False)
            except ValueError as e:
                if 'resize only works on single-segment arrays' in str(e):
                    self = self.copy()
                    self.resize(s)
                else:
                    raise
        elif other.shape[0] < self.shape[0]:
            N = other.shape[0]
            self.value[:-N] = self.value[N:]
        else:
            N = min(self.shape[0], other.shape[0])

        # if units are the same, can shortcut
        # NOTE: why not use isinstance here?
        if type(other) == type(self) and other.unit == self.unit:
            self.value[-N:] = other.value[-N:]
        # otherwise if its just a numpy array
        elif type(other) is type(self.value) or (
                other.dtype.name.startswith('uint')):
            self.value[-N:] = other[-N:]
        else:
            self[-N:] = other[-N:]
        try:
            self._xindex
        except AttributeError:
            if not resize:
                self.x0 = self.x0.value + other.shape[0] * self.dx.value
        else:
            if resize:
                try:
                    self.xindex.resize((s[0],), refcheck=False)
                except ValueError as exc:
                    if 'cannot resize' in str(exc):
                        self.xindex = self.xindex.copy()
                        self.xindex.resize((s[0],))
                    else:
                        raise
            else:
                self.xindex[:-other.shape[0]] = self.xindex[other.shape[0]:]
            try:
                self.xindex[-other.shape[0]:] = other._xindex
            except AttributeError:
                del self.xindex
                if not resize:
                    self.x0 = self.x0 + self.dx * other.shape[0]
            else:
                try:
                    self.dx = self.xindex[1] - self.xindex[0]
                except IndexError:
                    pass
                self.x0 = self.xindex[0]
        return self

    def prepend(self, other, gap='raise', inplace=True, pad=0, resize=True):
        """Connect another series onto the start of the current one.

        Parameters
        ----------
        other : `Series`
            another series of the same type as this one

        gap : `str`, optional, default: ``'raise'``
            action to perform if there's a gap between the other series
            and this one. One of

                - ``'raise'`` - raise an `Exception`
                - ``'ignore'`` - remove gap and join data
                - ``'pad'`` - pad gap with zeros

        inplace : `bool`, optional, default: `True`
            perform operation in-place, modifying current series,
            otherwise copy data and return new series

            .. warning::

               inplace prepend bypasses the reference check in
               `numpy.ndarray.resize`, so be carefully to only use this
               for arrays that haven't been sharing their memory!

        pad : `float`, optional, default: ``0.0``
            value with which to pad discontiguous `Series`
        resize : `bool`, optional, default: `True`

        Returns
        -------
        series : `TimeSeries`
            time-series containing joined data sets
        """
        out = other.append(self, gap=gap, inplace=False,
                           pad=pad, resize=resize)
        if inplace:
            self.resize(out.shape, refcheck=False)
            self[:] = out[:]
            self.x0 = out.x0.copy()
            del out
            return self
        else:
            return out

    def update(self, other, inplace=True):
        """Update this series by appending new data from an other
        and dropping the same amount of data off the start.

        This is a convenience method that just calls `~Series.append` with
        `resize=False`.
        """
        return self.append(other, inplace=inplace, resize=False)

    def crop(self, start=None, end=None, copy=False):
        """Crop this series to the given x-axis extent.

        Parameters
        ----------
        start : `float`, optional
            lower limit of x-axis to crop to, defaults to
            current `~Series.x0`

        end : `float`, optional
            upper limit of x-axis to crop to, defaults to current series end

        copy : `bool`, optional, default: `False`
            copy the input data to fresh memory, otherwise return a view

        Returns
        -------
        series : `Series`
            A new series with a sub-set of the input data

        Notes
        -----
        If either ``start`` or ``end`` are outside of the original
        `Series` span, warnings will be printed and the limits will
        be restricted to the :attr:`~Series.xspan`
        """
        # pin early starts to time-series start
        if start == self.xspan[0]:
            start = None
        elif start is not None and start < self.xspan[0]:
            warn('%s.crop given start smaller than current start, '
                 'crop will begin when the Series actually starts.'
                 % type(self).__name__)
            start = None
        # pin late ends to time-series end
        if end == self.xspan[1]:
            end = None
        if end is not None and end > self.xspan[1]:
            warn('%s.crop given end larger than current end, '
                 'crop will end when the Series actually ends.'
                 % type(self).__name__)
            end = None
        # find start index
        if start is None:
            idx0 = None
        else:
            idx0 = int(float(start - self.xspan[0]) // self.dx.value)
        # find end index
        if end is None:
            idx1 = None
        else:
            idx1 = int(float(end - self.xspan[0]) // self.dx.value)
            if idx1 >= self.size:
                idx1 = None
        # crop
        if copy:
            return self[idx0:idx1].copy()
        else:
            return self[idx0:idx1]

    def pad(self, pad_width, **kwargs):
        """Pad this series to a new size

        Parameters
        ----------
        pad_width : `int`, pair of `ints`
            number of samples by which to pad each end of the array.
            Single int to pad both ends by the same amount, or
            (before, after) `tuple` to give uneven padding
        **kwargs
            see :meth:`numpy.pad` for kwarg documentation

        Returns
        -------
        series : `Series`
            the padded version of the input

        See also
        --------
        numpy.pad
            for details on the underlying functionality
        """
        # format arguments
        kwargs.setdefault('mode', 'constant')
        if isinstance(pad_width, int):
            pad_width = (pad_width,)
        # form pad and view to this type
        new = numpy.pad(self, pad_width, **kwargs).view(type(self))
        # numpy.pad has stripped all metadata, so copy it over
        new.__metadata_finalize__(self)
        new._unit = self.unit
        # finally move the starting index based on the amount of left-padding
        new.x0 -= self.dx * pad_width[0]
        return new
