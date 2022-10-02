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
from math import floor

import numpy

from astropy.units import (Unit, Quantity, second, dimensionless_unscaled)
from astropy.io import registry as io_registry

from . import sliceutils
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
        new = super().__new__(cls, value, unit=unit, **kwargs)

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
        super().__array_finalize__(obj)
        # Array.__array_finalize__ might set _xindex to None, so delete it
        if getattr(self, '_xindex', None) is None:
            del self.xindex

    # -- series properties ----------------------

    def _update_index(self, axis, key, value):
        """Update the current axis index based on a given key or value

        This is an internal method designed to set the origin or step for
        an index, whilst updating existing Index arrays as appropriate

        Examples
        --------
        >>> self._update_index("x", "x0", 0)
        >>> self._update_index("x", "dx", 0)

        To actually set an index array, use `_set_index`
        """
        # delete current value if given None
        if value is None:
            return delattr(self, key)

        _key = "_{}".format(key)
        index = "{[0]}index".format(axis)
        unit = "{[0]}unit".format(axis)

        # convert float to Quantity
        if not isinstance(value, Quantity):
            try:
                value = Quantity(value, getattr(self, unit))
            except TypeError:
                value = Quantity(float(value), getattr(self, unit))

        # if value is changing, delete current index
        try:
            curr = getattr(self, _key)
        except AttributeError:
            delattr(self, index)
        else:
            if (
                    value is None
                    or getattr(self, key) is None
                    or not value.unit.is_equivalent(curr.unit)
                    or value != curr
            ):
                delattr(self, index)

        # set new value
        setattr(self, _key, value)
        return value

    def _set_index(self, key, index):
        """Set a new index array for this series
        """
        axis = key[0]

        # if given None, delete the current index
        if index is None:
            return delattr(self, key)

        origin = f"{axis}0"
        delta = f"d{axis}"

        # format input as an Index array
        if not isinstance(index, Index):
            try:
                unit = index.unit
            except AttributeError:
                unit = getattr(self, "_default_{}unit".format(axis))
            index = Index(index, unit=unit, copy=False)

        # reset other axis attributes
        if index.size:
            setattr(self, origin, index[0])
        if index.size > 1:
            if index.regular:  # delta will reset from index
                setattr(self, delta, index[1] - index[0])
            else:
                delattr(self, delta)

        # update index array
        setattr(self, "_{}".format(key), index)

    def _index_span(self, axis):
        from ..segments import Segment
        axisidx = ("x", "y", "z").index(axis)
        unit = getattr(self, "{}unit".format(axis))
        try:
            delta = getattr(self, "d{}".format(axis)).to(unit).value
        except AttributeError:  # irregular xindex
            index = getattr(self, "{}index".format(axis))
            try:
                delta = index.value[-1] - index.value[-2]
            except IndexError:
                raise ValueError("Cannot determine x-axis stride (dx)"
                                 "from a single data point")
            return Segment(index.value[0], index.value[-1] + delta)
        else:
            origin = getattr(self, "{}0".format(axis)).to(unit).value
            return Segment(origin, origin + self.shape[axisidx] * delta)

    # x0
    @property
    def x0(self):
        """X-axis coordinate of the first data point

        :type: `~astropy.units.Quantity` scalar
        """
        try:
            return self._x0
        except AttributeError:
            try:
                self._x0 = self._xindex[0]
            except (AttributeError, IndexError):
                self._x0 = Quantity(0, self.xunit)
            return self._x0

    @x0.setter
    def x0(self, value):
        self._update_index("x", "x0", value)

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
        self._update_index("x", "dx", value)

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
            self._xindex = Index.define(self.x0, self.dx, self.shape[0])
            return self._xindex

    @xindex.setter
    def xindex(self, index):
        self._set_index("xindex", index)

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
        return self._index_span("x")

    # -- series i/o -----------------------------

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data into a `Series`

        Arguments and keywords depend on the output format, see the
        online documentation for full details for each format, the
        parameters below are common to most formats.

        Parameters
        ----------
        source : `str`, `list`
            Source of data, any of the following:

            - `str` path of single data file,
            - `str` path of LAL-format cache file,
            - `list` of paths.

        *args
            Other arguments are (in general) specific to the given
            ``format``.

        format : `str`, optional
            Source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        **kwargs
            Other keywords are (in general) specific to the given ``format``.

        Returns
        -------
        data : `Series`

        Raises
        ------
        IndexError
            if ``source`` is an empty list

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

    # -- series plotting ------------------------

    def plot(self, method='plot', **kwargs):
        """Plot the data for this series

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            the newly created figure, with populated Axes.

        See also
        --------
        matplotlib.pyplot.figure
            for documentation of keyword arguments used to create the
            figure
        matplotlib.figure.Figure.add_subplot
            for documentation of keyword arguments used to create the
            axes
        matplotlib.axes.Axes.plot
            for documentation of keyword arguments used in rendering the data
        """
        from ..plot import Plot
        from ..plot.text import default_unit_label

        # correct for log scales and zeros
        if kwargs.get('xscale') == 'log' and self.x0.value == 0:
            kwargs.setdefault('xlim', (self.dx.value, self.xspan[1]))

        # make plot
        plot = Plot(self, method=method, **kwargs)

        # set default y-axis label (xlabel is set by Plot())
        default_unit_label(plot.gca().yaxis, self.unit)

        return plot

    def step(self, **kwargs):
        """Create a step plot of this series
        """
        where = kwargs.pop("where", "post")
        kwargs.setdefault(
            "drawstyle",
            "steps-{}".format(where),
        )
        data = self.append(self.value[-1:], inplace=False)
        return data.plot(**kwargs)

    # -- series methods -------------------------

    def shift(self, delta):
        """Shift this `Series` forward on the X-axis by ``delta``

        This modifies the series in-place.

        Parameters
        ----------
        delta : `float`, `~astropy.units.Quantity`, `str`
            The amount by which to shift (in x-axis units if `float`), give
            a negative value to shift backwards in time

        Examples
        --------
        >>> from gwpy.types import Series
        >>> a = Series([1, 2, 3, 4, 5], x0=0, dx=1, xunit='m')
        >>> print(a.x0)
        0.0 m
        >>> a.shift(5)
        >>> print(a.x0)
        5.0 m
        >>> a.shift('-1 km')
        -995.0 m
        """
        self.x0 = self.x0 + Quantity(delta, self.xunit)

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
        new = super().copy(order=order)
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

        See also
        --------
        numpy.diff
            for documentation on the underlying method
        """
        out = super().diff(n=n, axis=axis)
        try:
            out.x0 = self.x0 + self.dx * n
        except AttributeError:  # irregular xindex
            out.x0 = self.xindex[n]
        return out

    def __getslice__(self, i, j):
        new = super().__getslice__(i, j)
        if i:
            try:
                new.x0 = self.x0 + i * self.dx
            except AttributeError:  # irregular xindex
                new.x0 = self.xindex[i]
        return new

    def __getitem__(self, item):
        new = super().__getitem__(item)

        # slice axis 0 metadata
        slice_, = sliceutils.format_nd_slice(item, 1)
        if not sliceutils.null_slice(slice_):
            sliceutils.slice_axis_attributes(self, 'x', new, 'x', slice_)

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
            return self._is_compatible_gwpy(other)
        return self._is_compatible_numpy(other)

    def _compatibility_error(self, other, attr, name):
        return ValueError(
            "{} {} do not match: {} vs {}".format(
                type(self).__name__,
                name,
                getattr(self, attr, "none"),
                getattr(other, attr, "none")
            ),
        )

    def _compare_index(self, other, axis="x"):
        """Compare index attributes/arrays between self and other

        Raises
        ------
        ValueError
            if ``dx`` doesn't match, or ``xindex`` values are not present/are
            identical (as appropriate)
        """
        try:  # check step size, if possible
            _delta = "d{}".format(axis)
            deltaa = getattr(self, _delta)
            deltab = getattr(other, _delta)
            if deltaa != deltab:
                raise self._compatibility_error(
                    other,
                    _delta,
                    "{}-axis sample sizes".format(axis),
                )
        except AttributeError:  # irregular index
            _index = "_{}index".format(axis)
            idxa = getattr(self, _index, None)
            idxb = getattr(other, _index, None)
            if (
                idxa is None  # no index on 'self'
                or idxb is None  # no index on 'other'
                or not numpy.array_equal(idxa, idxb)  # indexes don't match
            ):
                raise self._compatibility_error(
                    other,
                    _index,
                    "{}-axis indexes".format(axis),
                )

    def _is_compatible_gwpy(self, other):
        """Check whether this series and another series are compatible
        """
        self._compare_index(other, axis="x")

        # check units
        if not (
            self.unit == other.unit
            or {self.unit, other.unit}.issubset(
                {dimensionless_unscaled, None},
            )
        ):
            raise self._compatibility_error(other, "unit", "units")

        # compatibility!
        return True

    def _is_compatible_numpy(self, other):
        """Check whether this series and a numpy.ndarray are compatible
        """
        arr = numpy.asarray(other)
        if arr.ndim != self.ndim:
            raise ValueError("Dimensionality does not match")
        if arr.dtype != self.dtype:
            warn("Array data types do not match: %s vs %s"
                 % (self.dtype, other.dtype))
        return True

    def append(self, other, inplace=True, pad=None, gap=None, resize=True):
        """Connect another series onto the end of the current one.

        Parameters
        ----------
        other : `Series`
            another series of the same type to connect to this one

        inplace : `bool`, optional
            perform operation in-place, modifying current series,
            otherwise copy data and return new series, default: `True`

            .. warning::

               `inplace` append bypasses the reference check in
               `numpy.ndarray.resize`, so be carefully to only use this
               for arrays that haven't been sharing their memory!

        pad : `float`, optional
            value with which to pad discontiguous series,
            by default gaps will result in a `ValueError`.

        gap : `str`, optional
            action to perform if there's a gap between the other series
            and this one. One of

            - ``'raise'`` - raise a `ValueError`
            - ``'ignore'`` - remove gap and join data
            - ``'pad'`` - pad gap with zeros

            If ``pad`` is given and is not `None`, the default is ``'pad'``,
            otherwise ``'raise'``. If ``gap='pad'`` is given, the default
            for ``pad`` is ``0``.

        resize : `bool`, optional
            resize this array to accommodate new data, otherwise shift the
            old data to the left (potentially falling off the start) and
            put the new data in at the end, default: `True`.

        Returns
        -------
        series : `Series`
            a new series containing joined data sets
        """
        if gap is None:
            gap = 'raise' if pad is None else 'pad'
        if pad is None and gap == 'pad':
            pad = 0.

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
        elif type(other) is type(self.value) or (  # noqa: E721
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

    def prepend(self, other, inplace=True, pad=None, gap=None, resize=True):
        """Connect another series onto the start of the current one.

        Parameters
        ----------
        other : `Series`
            another series of the same type as this one

        inplace : `bool`, optional
            perform operation in-place, modifying current series,
            otherwise copy data and return new series, default: `True`

            .. warning::

               `inplace` prepend bypasses the reference check in
               `numpy.ndarray.resize`, so be carefully to only use this
               for arrays that haven't been sharing their memory!

        pad : `float`, optional
            value with which to pad discontiguous series,
            by default gaps will result in a `ValueError`.

        gap : `str`, optional
            action to perform if there's a gap between the other series
            and this one. One of

            - ``'raise'`` - raise a `ValueError`
            - ``'ignore'`` - remove gap and join data
            - ``'pad'`` - pad gap with zeros

            If `pad` is given and is not `None`, the default is ``'pad'``,
            otherwise ``'raise'``.

        resize : `bool`, optional
            resize this array to accommodate new data, otherwise shift the
            old data to the left (potentially falling off the start) and
            put the new data in at the end, default: `True`.

        Returns
        -------
        series : `TimeSeries`
            time-series containing joined data sets
        """
        out = other.append(self, inplace=False, gap=gap, pad=pad,
                           resize=resize)
        if inplace:
            self.resize(out.shape, refcheck=False)
            self[:] = out[:]
            self.x0 = out.x0.copy()
            del out
            return self
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
        x0, x1 = self.xspan
        xtype = type(x0)
        if isinstance(start, Quantity):
            start = start.to(self.xunit).value
        if isinstance(end, Quantity):
            end = end.to(self.xunit).value

        # pin early starts to time-series start
        if start == x0:
            start = None
        elif start is not None and xtype(start) < x0:
            warn('%s.crop given start smaller than current start, '
                 'crop will begin when the Series actually starts.'
                 % type(self).__name__)
            start = None

        # pin late ends to time-series end
        if end == x1:
            end = None
        if end is not None and xtype(end) > x1:
            warn('%s.crop given end larger than current end, '
                 'crop will end when the Series actually ends.'
                 % type(self).__name__)
            end = None

        # check if series is irregular
        try:
            self.dx
        except AttributeError:
            irregular = True
        else:
            irregular = False

        # find start index
        if start is None:
            idx0 = None
        else:
            if not irregular:
                idx0 = int((xtype(start) - x0) // self.dx.value)
            else:
                idx0 = numpy.searchsorted(
                    self.xindex.value, xtype(start), side="left"
                )

        # find end index
        if end is None:
            idx1 = None
        else:
            if not irregular:
                idx1 = int((xtype(end) - x0) // self.dx.value)
                if idx1 >= self.size:
                    idx1 = None
            else:
                if xtype(end) >= self.xindex.value[-1]:
                    idx1 = None
                else:
                    idx1 = (
                        numpy.searchsorted(
                            self.xindex.value, xtype(end), side="left"
                        )
                    )

        # crop
        if copy:
            return self[idx0:idx1].copy()
        return self[idx0:idx1]

    def pad(self, pad_width, **kwargs):
        """Pad this series to a new size

        Parameters
        ----------
        pad_width : `int`, pair of `ints`
            number of samples by which to pad each end of the array;
            given a single `int` to pad both ends by the same amount,
            or a (before, after) `tuple` for assymetric padding

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
        new = numpy.pad(self.value, pad_width, **kwargs).view(type(self))
        # numpy.pad has stripped all metadata, so copy it over
        new.__metadata_finalize__(self)
        new._unit = self.unit
        # finally move the starting index based on the amount of left-padding
        new.x0 = new.x0 - self.dx * pad_width[0]
        return new

    def inject(self, other):
        """Add two compatible `Series` along their shared x-axis values.

        Parameters
        ----------
        other : `Series`
            a `Series` whose xindex intersects with `self.xindex`

        Returns
        -------
        out : `Series`
            the sum of `self` and `other` along their shared x-axis values

        Raises
        ------
        ValueError
            if `self` and `other` have incompatible units or xindex intervals

        Notes
        -----
        If `other.xindex` and `self.xindex` do not intersect, this method will
        return a copy of `self`. If the series have uniformly offset indices,
        this method will raise a warning.

        If `self.xindex` is an array of timestamps, and if `other.xspan` is
        not a subset of `self.xspan`, then `other` will be cropped before
        being adding to `self`.

        Users who wish to taper or window their `Series` should do so before
        passing it to this method. See :meth:`TimeSeries.taper` and
        :func:`~gwpy.signal.window.planck` for more information.
        """
        # check Series compatibility
        self.is_compatible(other)
        if (self.xunit == second) and (other.xspan[0] < self.xspan[0]):
            other = other.crop(start=self.xspan[0])
        if (self.xunit == second) and (other.xspan[1] > self.xspan[1]):
            other = other.crop(end=self.xspan[1])
        ox0 = other.x0.to(self.x0.unit)
        idx = ((ox0 - self.x0) / self.dx).value
        if not idx.is_integer():
            warn('Series have overlapping xspan but their x-axis values are '
                 'uniformly offset. Returning a copy of the original Series.')
            return self.copy()
        # add the Series along their shared samples
        slice_ = slice(int(idx), int(idx) + other.size)
        out = self.copy()
        out.value[slice_] += other.value
        return out
