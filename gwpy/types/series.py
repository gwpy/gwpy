# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""The `Series` is a one-dimensional array with metadata."""

from __future__ import annotations

import contextlib
from math import floor
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)
from warnings import warn

import numpy
from astropy.units import (
    Quantity,
    Unit,
    second,
)

from ..io.registry import UnifiedReadWriteMethod
from . import sliceutils
from .array import Array
from .connect import (
    SeriesRead,
    SeriesWrite,
)
from .index import Index

if TYPE_CHECKING:
    from typing import (
        ClassVar,
        Literal,
        Self,
    )

    from astropy.units import UnitBase
    from astropy.units.typing import QuantityLike

    from ..plot import Plot
    from ..segments import Segment
    from ..typing import UnitLike
    from .sliceutils import SliceLike

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class Series(Array):
    """A one-dimensional data series.

    A `Series` is defined as an array of data indexed upon an axis, meaning
    each sample maps to a position upon the axis. By convention the X axis
    is used to define the index, with the `~Series.x0`, `~Series.dx`, and
    `~Series.xindex` attributes allowing the positions of the data to be
    well defined.

    Parameters
    ----------
    value : array-like
        Input data array.

    unit : `~astropy.units.Unit`, optional
        Physical unit of these data.

    x0 : `float`, `~astropy.units.Quantity`, optional, default: `0`
        The starting value for the x-axis of this array.

    dx : `float`, `~astropy.units.Quantity, optional, default: `1`
        The step size for the x-axis of this array.

    xindex : `array-like`
        The complete array of x-axis values for this array.
        This argument takes precedence over `x0` and `dx` so should be
        given in place of these if relevant, not alongside.

    xunit : `~astropy.units.Unit`, optional
        The unit of the x-axis index. If not given explicitly, it will be
        taken from any of `dx`, `x0`, or `xindex`, or set to a boring default.

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine.

    name : `str`, optional
        Descriptive title for this array.

    channel : `~gwpy.detector.Channel`, `str`, optional
        Source data stream for these data.

    dtype : `~numpy.dtype`, optional
        Input data type.

    copy : `bool`, optional, default: `False`
        Choose to copy the input data to new memory.

    subok : `bool`, optional, default: `True`
        Allow passing of sub-classes by the array generator.

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

    _metadata_slots: ClassVar[tuple[str, ...]] = (
        *Array._metadata_slots,
        "x0",
        "dx",
        "xindex",
    )
    _default_xunit: ClassVar[UnitBase] = Unit("")
    _ndim: ClassVar[int] = 1

    def __new__(
        cls,
        value: QuantityLike,
        unit: UnitBase | str | None = None,
        x0: QuantityLike | None = None,
        dx: QuantityLike | None = None,
        xindex: QuantityLike | None = None,
        xunit: UnitBase | str | None = None,
        **kwargs,
    ) -> Self:
        """Create a new `Series."""
        # check input data dimensions are OK
        shape = numpy.shape(value)
        if len(shape) != cls._ndim:
            msg = (
                f"cannot generate {cls.__name__} with "
                f"{len(shape)}-dimensional data"
            )
            raise ValueError(msg)

        # create new object
        new = super().__new__(
            cls,
            value,
            unit=unit,
            **kwargs,
        )

        # set x-axis metadata from xindex
        if xindex is not None:

            if len(xindex) != len(value):
                msg = "xindex must have the same length as data"
                raise ValueError(msg)

            # warn about duplicate settings
            if dx is not None:
                warn(
                    f"xindex was given to {cls.__name__}(), "
                    "dx will be ignored",
                    stacklevel=2,
                )
            if x0 is not None:
                warn(
                    f"xindex was given to {cls.__name__}(), "
                    "x0 will be ignored",
                    stacklevel=2,
                )
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

    # -- series creation -------------

    def __array_finalize__(self, obj: Self | None) -> None:
        """Finalize the array after creation."""
        super().__array_finalize__(obj)

        # Array.__array_finalize__ might set _xindex to None, so delete it
        if getattr(self, "_xindex", None) is None:
            del self.xindex

    # -- series properties -----------

    def _update_index(
        self,
        axis: str,
        attr: str,
        value: QuantityLike,
    ) -> None:
        """Update the current axis index based on a given attr or value.

        This is an internal method designed to set the origin or step for
        an index, whilst updating existing Index arrays as appropriate.

        Parameters
        ----------
        axis : `str`
            The name of the axis to update, e.g. ``"x"``.

        attr : `str`
            The name of the attribute to set.

        value : `Quantity`, `float`
            The value to set for ``attr``.

        Examples
        --------
        >>> self._update_index("x", "x0", 0)
        >>> self._update_index("x", "dx", 1)

        Notes
        -----
        To actually set an index array, use `_set_index`.
        """
        index = f"{axis[0]}index"

        # delete current value if given None
        if value is None:
            delattr(self, attr)
            delattr(self, index)
            return

        _attr = f"_{attr}"
        unit = f"{axis[0]}unit"

        # convert float to Quantity
        if not isinstance(value, Quantity):
            try:
                value = Quantity(value, getattr(self, unit))
            except TypeError:
                value = Quantity(float(value), getattr(self, unit))

        # if value is changing, delete current index
        try:
            curr = getattr(self, _attr)
        except AttributeError:
            delattr(self, index)
        else:
            if (
                value is None
                or getattr(self, attr) is None
                or not value.unit.is_equivalent(curr.unit)
                or value != curr
            ):
                delattr(self, index)

        # set new value
        setattr(self, _attr, value)

    def _set_index(
        self,
        axis: Literal["x", "y", "z"],
        index: QuantityLike,
    ) -> None:
        """Set a new index array for this series.

        Parameters
        ----------
        axis : `str`
            The name of the axis to update, e.g. ``"x"``.

        index : `Quantity`, `numpy.ndarray`
            The index array to apply.

        Examples
        --------
        >>> self._set_index("x", numpy.arange(100))
        """
        attr = f"{axis}index"

        # if given None, delete the current index
        if index is None:
            delattr(self, attr)
            return

        origin = f"{axis}0"
        delta = f"d{axis}"

        # format input as an Index array
        if not isinstance(index, Index):
            try:
                unit = index.unit
            except AttributeError:
                unit = getattr(self, f"_default_{axis}unit")
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
        setattr(self, f"_{attr}", index)

    def _index_span(self, axis: str) -> Segment:
        """Return the span of the given axis.

        Parameters
        ----------
        axis : `str`
            One of 'x', 'y', or 'z'.

        Returns
        -------
        span : `~gwpy.segments.Segment`
            The ``[start, stop)`` span of this series.
        """
        from ..segments import Segment

        axisidx = ("x", "y", "z").index(axis)
        unit = getattr(self, f"{axis}unit")

        try:
            delta = getattr(self, f"d{axis}").to(unit).value
        except AttributeError:  # irregular xindex
            index = getattr(self, f"{axis}index")
            try:
                delta = index.value[-1] - index.value[-2]
            except IndexError as exc:
                msg = (
                    "cannot determine x-axis stride (dx)"
                    "from a single data point"
                )
                raise ValueError(msg) from exc
            return Segment(index.value[0], index.value[-1] + delta)

        origin = getattr(self, f"{axis}0").to(unit).value
        return Segment(origin, origin + self.shape[axisidx] * delta)

    # x0
    @property
    def x0(self) -> Quantity:
        """X-axis coordinate of the first data point."""
        self._x0: Quantity
        try:
            return self._x0
        except AttributeError:
            try:
                self._x0 = self._xindex[0]
            except (AttributeError, IndexError):
                self._x0 = Quantity(0, self.xunit)
            return self._x0

    @x0.setter
    def x0(self, value: QuantityLike) -> None:
        self._update_index("x", "x0", value)

    @x0.deleter
    def x0(self) -> None:
        with contextlib.suppress(AttributeError):
            del self._x0

    # dx
    @property
    def dx(self) -> Quantity:
        """X-axis sample separation."""
        self._dx: Quantity
        try:
            return self._dx
        except AttributeError:
            try:
                self._xindex  # noqa: B018
            except AttributeError:
                self._dx = Quantity(1, self.xunit)
            else:
                if not self.xindex.regular:
                    msg = (
                        "this series has an irregular x-axis index, "
                        "so 'dx' is not well defined"
                    )
                    raise AttributeError(msg)
                self._dx = self.xindex[1] - self.xindex[0]
            return self._dx

    @dx.setter
    def dx(self, value: QuantityLike) -> None:
        self._update_index("x", "dx", value)

    @dx.deleter
    def dx(self) -> None:
        with contextlib.suppress(AttributeError):
            del self._dx

    # xindex
    @property
    def xindex(self) -> Index:
        """Positions of the data on the x-axis."""
        self._xindex: Index
        try:
            return self._xindex
        except AttributeError:
            self._xindex = Index.define(self.x0, self.dx, self.shape[0])
            return self._xindex

    @xindex.setter
    def xindex(self, index: QuantityLike) -> None:
        self._set_index("x", index)

    @xindex.deleter
    def xindex(self) -> None:
        with contextlib.suppress(AttributeError):
            del self._xindex

    # xunit
    @property
    def xunit(self) -> UnitBase:
        """Unit of x-axis index."""
        try:
            return self._dx.unit
        except AttributeError:
            try:
                return self._x0.unit
            except AttributeError:
                return self._default_xunit

    @xunit.setter
    def xunit(self, unit: UnitLike) -> None:
        unit = Unit(unit)
        try:  # set the index, if present
            self.xindex = self._xindex.to(unit)
        except AttributeError:  # or just set the start and step
            self.dx = self.dx.to(unit)
            self.x0 = self.x0.to(unit)

    @property
    def xspan(self) -> Segment:
        """X-axis [low, high) segment encompassed by these data."""
        return self._index_span("x")

    # -- series i/o ------------------

    read = UnifiedReadWriteMethod(SeriesRead)
    write = UnifiedReadWriteMethod(SeriesWrite)

    # -- series plotting -------------

    def plot(
        self,
        method: str = "plot",
        **kwargs,
    ) -> Plot:
        """Plot the data for this series.

        Parameters
        ----------
        method : `str`, optional
            The method on the `~gwpy.plot.Axes` to call to render
            this object. Default is ``"plot"`` (`~gwpy.plot.Axes.plot`).

        kwargs
            Other keyword arguments are passed to the relevant
            Axes plotting method.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            The newly created figure, with populated Axes.

        See Also
        --------
        matplotlib.pyplot.figure
            For documentation of keyword arguments used to create the figure.

        matplotlib.figure.Figure.add_subplot
            For documentation of keyword arguments used to create the axes.

        matplotlib.axes.Axes.plot
            For documentation of keyword arguments used in rendering the data.
        """
        from ..plot import Plot
        from ..plot.text import default_unit_label

        # correct for log scales and zeros
        if kwargs.get("xscale") == "log" and self.x0.value == 0:
            kwargs.setdefault("xlim", (self.dx.value, self.xspan[1]))

        # make plot
        plot = Plot(self, method=method, **kwargs)

        # set default y-axis label (xlabel is set by Plot())
        default_unit_label(plot.gca().yaxis, self.unit)

        return plot

    def step(self, **kwargs) -> Plot:
        """Create a step plot of this series.

        kwargs
            All keyword arguments are passed to the :meth:`plot` method.
            of this series.

        See Also
        --------
        plot
            For details of the plotting.
        """
        where = kwargs.pop("where", "post")
        kwargs.setdefault(
            "drawstyle",
            f"steps-{where}",
        )
        data = self.append(self.value[-1:], inplace=False)
        return data.plot(**kwargs)

    # -- series methods --------------

    def shift(self, delta: QuantityLike) -> None:
        """Shift this `Series` forward on the X-axis by ``delta``.

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

    def value_at(self, x: QuantityLike) -> Quantity:
        """Return the value of this `Series` at the given `xindex` value.

        Parameters
        ----------
        x : `float`, `~astropy.units.Quantity`
            The `xindex` value at which to search.

        Returns
        -------
        y : `~astropy.units.Quantity`
            The value of this Series at the given `xindex` value.

        Raises
        ------
        IndexError
            If ``x`` doesn't match an X-index value.
        """
        x = Quantity(x, self.xindex.unit).value
        try:
            idx = (self.xindex.value == x).nonzero()[0][0]
        except IndexError as e:
            e.args = (f"Value {x!r} not found in array index",)
            raise
        return self[idx]

    def copy(self, order: Literal["C", "F", "A", "K"] = "C") -> Self:
        """Return a copy of this `Series`.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            The desired memory layout order for the copy.

        See Also
        --------
        numpy.ndarray.copy
            For details of the copy operation.
        """
        new = super().copy(order=order)
        with contextlib.suppress(AttributeError):
            new._xindex = self._xindex.copy()
        return new

    copy.__doc__ = Array.copy.__doc__

    def zip(self) -> numpy.ndarray:
        """Zip the `xindex` and `value` arrays of this `Series`.

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

    def diff(self, n: int = 1, axis: int = -1) -> Self:
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
            For documentation on the underlying method.
        """
        out = super().diff(n=n, axis=axis)
        try:
            out.x0 = self.x0 + self.dx * n
        except AttributeError:  # irregular xindex
            out.x0 = self.xindex[n]
        return out

    @overload
    def __getitem__(self, item: int) -> Quantity: ...
    @overload
    def __getitem__(self, item: SliceLike | tuple[SliceLike, ...]) -> Self: ...

    def __getitem__(
        self,
        item: SliceLike | tuple[SliceLike, ...],
    ) -> Self | Quantity:
        """Get an item, or a slice, from this `Series`."""
        new = super().__getitem__(item)

        # slice axis 0 metadata
        (slice_,) = sliceutils.format_nd_slice(item, 1)
        if not sliceutils.null_slice(slice_):
            sliceutils.slice_axis_attributes(self, "x", new, "x", slice_)

        return new

    def is_contiguous(
        self,
        other: Series | numpy.ndarray | list,
        tol: float = 2 ** -18,
    ) -> int:
        """Check whether other is contiguous with self.

        Parameters
        ----------
        other : `Series`, `numpy.ndarray`
            Another series of the same type to test for contiguity.

        tol : `float`, optional
            The numerical tolerance of the test.

        Returns
        -------
        1
            If ``other`` is contiguous with this series,
            i.e. would attach seamlessly onto the end.

        -1
            If ``other`` is anti-contiguous with this seires,
            i.e. would attach seamlessly onto the start.

        0
            If ``other`` is completely dis-contiguous with this series.

        Notes
        -----
        If ``other`` is an array that doesn't have index information (e.g.
        a `numpy.ndarray`), this method always returns ``1``.

        If ``self`` ***or*** ``other``` have an irregular `Index` array
        (e.g. aren't linearly sampled), this method will always return ``1``
        if ``other`` starts after ``self`` finishes, or ``-1``` if the inverse.
        If the two arrays overlap, that is bad and will raise an error.
        """
        self.is_compatible(other)
        if isinstance(other, type(self)):
            if abs(float(self.xspan[1] - other.xspan[0])) < tol:
                return 1
            if abs(float(other.xspan[1] - self.xspan[0])) < tol:
                return -1
            return 0
        if type(other) in [list, tuple, numpy.ndarray]:
            return 1
        return 0

    def is_compatible(self, other: list | numpy.ndarray) -> bool:
        """Check whether this series and other have compatible metadata.

        This method tests that the `sample size <Series.dx>`, and the
        `~Series.unit` match.
        """
        self.check_compatible(other)
        # we survived!
        return True

    def check_compatible(
        self,
        other: list | numpy.ndarray,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] | None = "safe",
        *,
        irregular_equal: bool = True,
    ) -> None:
        """Check whether this Series and ``other`` are compatible.

        Parameters
        ----------
        other : `numpy.ndarray`, `Series`
            The array to compare to.

        casting : `str`, optional
            The type of casting to support when comparing dtypes.

        irregular_equal : `bool`, optional
            Require irregular indices to be equal (default).
            If ``irregular_equal=False`` and either (or both) of the series
            are irregular, this method just returns without doing anything.

        Raises
        ------
        ValueError
            If any metadata elements aren't compatible.

        TypeError
            If the dtype can't be safely cast between the arrays.
        """
        if isinstance(other, type(self)):
            self._check_compatible_gwpy(
                other,
                irregular_equal=irregular_equal,
            )
        if isinstance(other, Quantity):
            self._check_compatible_quantity(other)
        self._check_compatible_numpy(other, casting=casting)

    def _compatibility_error(
        self,
        other: Series,
        attr: str,
        desc: str,
    ) -> ValueError:
        """Construct a `ValueError` to be raised somewhere else."""
        msg = (
            f"{type(self).__name__} {desc} do not match: "
            f"{getattr(self, attr, None)} vs "
            f"{getattr(other, attr, None)}"
        )
        return ValueError(msg)

    def _check_compatible_index(
        self,
        other: Series,
        axis: str = "x",
        *,
        irregular_equal: bool = True,
    ) -> None:
        """Compare index attributes/arrays between self and other.

        Parameters
        ----------
        other : `Series`
            The series to compare to this one.

        axis : `str`, optional
            The Axis index to compare.
            Default is ``x``.

        irregular_equal : `bool`, optional
            Require irregular indices to be equal (default).
            If ``irregular_equal=False`` and either (or both) of the series
            are irregular, this method just returns without doing anything.

        Raises
        ------
        ValueError
            If ``dx`` doesn't match, or ``xindex`` values are not present/are
            identical (as appropriate).
        """
        _delta = f"d{axis}"
        try:  # check step size, if possible
            deltaa = getattr(self, _delta)
            deltab = getattr(other, _delta)
        except AttributeError:  # irregular index
            # at least one of the series is irregular
            _index = f"_{axis}index"
            idxa = getattr(self, _index, None)
            idxb = getattr(other, _index, None)

            # compare the indices
            if irregular_equal and (
                idxa is None  # no index on 'self'
                or idxb is None  # no index on 'other'
                or not numpy.array_equal(idxa, idxb)  # indexes don't match
            ):
                raise self._compatibility_error(
                    other,
                    _index,
                    f"{axis}-axis indexes",
                ) from None
        else:
            # both series are regular, check that the step sizes match
            if deltaa != deltab:
                raise self._compatibility_error(
                    other,
                    _delta,
                    f"{axis}-axis sample sizes",
                )

    def _check_compatible_gwpy(
        self,
        other: Quantity,
        *,
        irregular_equal: bool = True,
    ) -> None:
        """Check whether this series and another series are compatible.

        This method checks that the Index arrays are compatible.
        """
        self._check_compatible_index(
            other,
            axis="x",
            irregular_equal=irregular_equal,
        )

    def _check_compatible_quantity(
        self,
        other: Quantity,
    ) -> None:
        """Check with this Series and another `Quantity` are compatible.

        This method checks that the units match.

        Raises
        ------
        ValueError
            If this series and ``other`` are incompatible.

        See Also
        --------
        astropy.units.Unit.is_equivalent
            For details of the unit compatibility check.
        """
        # if neither quantity has a unit, that's fine
        if self.unit is None and other.unit is None:
            return
        # check that the units are equivalent
        if (
            self.unit is None
            or other.unit is None
            or not self.unit.is_equivalent(other.unit)
        ):
            raise self._compatibility_error(other, "unit", "units")

    def _check_compatible_numpy(
        self,
        other: list | numpy.ndarray,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] | None = "safe",
    ) -> None:
        """Check whether this series and a numpy.ndarray are compatible.

        This method checks that the dimensions are the same, and that
        the dtypes match.

        Raises
        ------
        ValueError
            If the dimensions don't match.

        TypeError
            If the data types can't be cast safely.

        See Also
        --------
        numpy.can_cast
            For details of the dtype check.
        """
        arr = numpy.asarray(other)
        if arr.ndim != self.ndim:
            msg = "dimensionality does not match"
            raise ValueError(msg)
        if not numpy.can_cast(arr.dtype, self.dtype, casting=casting):
            msg = f"array data types do not match: {self.dtype} vs {arr.dtype}"
            raise TypeError(msg)

    def append(
        self,
        other: numpy.ndarray | Series,
        *,
        inplace: bool = True,
        gap: Literal["raise", "ignore", "pad"] | None = None,
        pad: float | None = None,
        resize: bool = True,
    ) -> Self:
        """Connect another series onto this one.

        Parameters
        ----------
        other : `numpy.ndarray`, `Series`
            Another `Series`, or a simple data array to connect to this one.

        inplace : `bool`, optional
            If `True` (default) perform the operation in-place, modifying current
            series.
            If `False` copy the data to new memory before modifying.

            .. warning::

               ``inplace`` append bypasses the reference check in
               `numpy.ndarray.resize`, so be carefully to only use this
               for arrays that haven't been sharing their memory!

        gap : `str`, optional
            Action to perform if there's a gap between the other series
            and this one. One of

            - ``'raise'`` - raise a `ValueError`
            - ``'ignore'`` - remove gap and join data
            - ``'pad'`` - pad gap with zeros

            If ``pad`` is given and is not `None`, the default is
            ``gap='pad'``, otherwise ``gap='raise'``.

            If ``gap='pad'`` is given, the default for ``pad`` is ``0``.

        pad : `float`, optional
            Value with which to pad discontiguous series,
            by default gaps will result in a `ValueError`.

        resize : `bool`, optional
            If `True` (default) resize this array to accommodate new data.
            If `False` roll the current data like a buffer to the left
            and insert new data at the other end.

        Returns
        -------
        series : `Series`
            A new series containing joined data sets.
        """
        if gap is None:
            gap = "raise" if pad is None else "pad"
        if pad is None and gap == "pad":
            pad = 0.0

        # check metadata
        self.is_compatible(other)
        # make copy if needed
        if not inplace:
            self = self.copy()
        # fill gap
        if self.is_contiguous(other) != 1:
            other = cast("Self", other)
            if gap == "pad":
                pad = cast("float", pad)
                ngap = floor((other.xspan[0] - self.xspan[1]) / self.dx.value + 0.5)
                if ngap < 1:
                    msg = (
                        "Cannot append {0} that starts before this one:\n"
                        "    {0} 1 span: {1}\n    {0} 2 span: {2}".format(
                            type(self).__name__,
                            self.xspan,
                            other.xspan,
                        )
                    )
                    raise ValueError(
                        msg,
                    )
                gapshape = list(self.shape)
                gapshape[0] = int(ngap)
                padding = (numpy.ones(gapshape) * pad).astype(self.dtype)
                self.append(padding, inplace=True, resize=resize)
            elif gap == "ignore":
                pass
            elif self.xspan[0] < other.xspan[0] < self.xspan[1]:
                msg = (
                    "Cannot append overlapping {0}s:\n"
                    "    {0} 1 span: {1}\n    {0} 2 span: {2}".format(
                        type(self).__name__,
                        self.xspan,
                        other.xspan,
                    )
                )
                raise ValueError(
                    msg,
                )
            else:
                msg = (
                    "Cannot append discontiguous {0}\n"
                    "    {0} 1 span: {1}\n    {0} 2 span: {2}".format(
                        type(self).__name__,
                        self.xspan,
                        other.xspan,
                    )
                )
                raise ValueError(
                    msg,
                )

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
                if "resize only works on single-segment arrays" in str(e):
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
        if isinstance(other, Series) and other.unit == self.unit:
            self.value[-N:] = other.value[-N:]
        # otherwise if its just a numpy array
        elif type(other) is type(self.value) or (other.dtype.name.startswith("uint")):
            self.value[-N:] = other[-N:]
        else:
            self[-N:] = other[-N:]
        try:
            self._xindex  # noqa: B018
        except AttributeError:
            if not resize:
                self.x0 = self.x0.value + other.shape[0] * self.dx.value
        else:
            if resize:
                try:
                    self.xindex.resize((s[0],), refcheck=False)
                except ValueError as exc:
                    if "cannot resize" in str(exc):
                        self.xindex = self.xindex.copy()
                        self.xindex.resize((s[0],))
                    else:
                        raise
            else:
                self.xindex[: -other.shape[0]] = self.xindex[other.shape[0] :]
            try:
                self.xindex[-other.shape[0] :] = other._xindex  # type: ignore[union-attr]  # noqa: SLF001
            except AttributeError:
                del self.xindex
                if not resize:
                    self.x0 = self.x0 + self.dx * other.shape[0]
            else:
                with contextlib.suppress(IndexError):
                    self.dx = self.xindex[1] - self.xindex[0]
                self.x0 = self.xindex[0]
        return self

    def prepend(
        self,
        other: QuantityLike,
        *,
        inplace: bool = True,
        gap: Literal["raise", "ignore", "pad"] | None = None,
        pad: float | None = None,
        resize: bool = True,
    ) -> Series:
        """Connect another series onto the start of the current one.

        Parameters
        ----------
        other : `numpy.ndarray`, `Series`
            The data to prepend to this series.

        inplace : `bool`, optional
            If `True` (default) perform the operation in-place, modifying current
            series.
            If `False` copy the data to new memory before modifying.

            .. warning::

               ``inplace`` append bypasses the reference check in
               `numpy.ndarray.resize`, so be carefully to only use this
               for arrays that haven't been sharing their memory!

        gap : `str`, optional
            Action to perform if there's a gap between the other series
            and this one. One of

            - ``'raise'`` - raise a `ValueError`
            - ``'ignore'`` - remove gap and join data
            - ``'pad'`` - pad gap with zeros

            If ``pad`` is given and is not `None`, the default is
            ``gap='pad'``, otherwise ``gap='raise'``.

            If ``gap='pad'`` is given, the default for ``pad`` is ``0``.

        pad : `float`, optional
            Value with which to pad discontiguous series,
            by default gaps will result in a `ValueError`.

        resize : `bool`, optional
            If `True` (default) resize this array to accommodate new data.
            If `False` roll the current data like a buffer to the left or right
            (depending on ``prepend``) and insert new data at the other end.

        Returns
        -------
        series : `Series`
            The modified series.
        """
        out = other.append(self, inplace=False, gap=gap, pad=pad, resize=resize)
        if inplace:
            self.resize(out.shape, refcheck=False)
            self[:] = out[:]
            self.x0 = out.x0.copy()
            del out
            return self
        return out

    def update(
        self,
        other: QuantityLike,
        *,
        inplace: bool = True,
        gap: Literal["raise", "ignore", "pad"] | None = None,
        pad: float | None = None,
    ) -> Self:
        """Update this series by appending new data like a buffer.

        Old data (at the start) are dropped to maintain a fixed size.

        This is a convenience method that just calls `~Series.append` with
        ``resize=False``.

        Parameters
        ----------
        other : `Series`, `numpy.ndarray`
            The data to add to the end of this `Series`.

        inplace : `bool`
            If `True` (default) modify the data in place.
            If `False` copy the data to new memory.

        gap : `str`, optional
            Action to perform if there's a gap between the other series
            and this one. One of

            - ``'raise'`` - raise a `ValueError`
            - ``'ignore'`` - remove gap and join data
            - ``'pad'`` - pad gap with zeros

            If ``pad`` is given and is not `None`, the default is
            ``gap='pad'``, otherwise ``gap='raise'``.

            If ``gap='pad'`` is given, the default for ``pad`` is ``0``.

        pad : `float`, optional
            Value with which to pad discontiguous series,
            by default gaps will result in a `ValueError`.

        Returns
        -------
        series : `Series`
            Either the same series (if ``inplace=True``) or a new
            series (if ``inplace=False``) with ``other`` data added
            to the end of this 'buffer'.

        See Also
        --------
        append
            For details of the data manipulation.
        """
        return self.append(
            other,
            inplace=inplace,
            gap=gap,
            pad=pad,
            resize=False,
        )

    def crop(
        self,
        start: Quantity | float | None = None,
        end: Quantity | float | None = None,
        *,
        copy: bool = False,
    ) -> Self:
        """Crop this series to the given x-axis extent.

        Parameters
        ----------
        start : `float`, optional
            Lower limit of x-axis to crop to, defaults to
            :attr:`~Series.x0`.

        end : `float`, optional
            Upper limit of x-axis to crop to, defaults to series end.

        copy : `bool`, optional
            Copy the input data to fresh memory,
            otherwise return a view (default).

        Returns
        -------
        series : `Series`
            A new series with a sub-set of the input data.

        Notes
        -----
        If either ``start`` or ``end`` are outside of the original
        `Series` span, warnings will be printed and the limits will
        be restricted to the :attr:`~Series.xspan`.
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
            warn(
                f"{type(self).__name__}.crop given start smaller than current "
                "start, crop will begin when the Series actually starts.",
                stacklevel=2,
            )
            start = None

        # pin late ends to time-series end
        if end == x1:
            end = None
        if end is not None and xtype(end) > x1:
            warn(
                f"{type(self).__name__}.crop given end larger than current "
                "end, crop will begin when the Series actually ends.",
                stacklevel=2,
            )
            end = None

        # check if we have an index to use when searching
        have_xindex = getattr(self, "_xindex", None) is not None

        # find start index
        if start is None:
            idx0 = None
        elif have_xindex:
            idx0 = numpy.searchsorted(
                self.xindex.value,
                xtype(start),
                side="left",
            )
        else:
            idx0 = floor((xtype(start) - x0) / self.dx.value)

        # find end index
        if end is None:
            idx1 = None
        elif have_xindex:
            idx1 = numpy.searchsorted(
                self.xindex.value,
                xtype(end),
                side="left",
            )
        else:
            idx1 = floor((xtype(end) - x0) / self.dx.value)

        # crop
        if copy:
            return self[idx0:idx1].copy()
        return self[idx0:idx1]

    def pad(
        self,
        pad_width: int | tuple[int, int],
        **kwargs,
    ) -> Self:
        """Pad this series to a new size.

        This just wraps `numpy.pad` and handles shifting the `Index` to
        accommodate padding on the left.

        Parameters
        ----------
        pad_width : `int`, `tuple[int, int]`
            Number of samples by which to pad each end of the array;
            given a single `int` to pad both ends by the same amount,
            or a ``(before, after)`` `tuple` for assymetric padding.

        kwargs
            Other keyword arguments are passed to `numpy.pad`.

        Returns
        -------
        series : `Series`
            The padded version of the input.

        See Also
        --------
        numpy.pad
            For details on the pad function and valid keyword arguments.
        """
        # format arguments
        kwargs.setdefault("mode", "constant")
        if isinstance(pad_width, int):
            pad_width = cast("tuple[int, int]", (pad_width,))
        # form pad and view to this type
        new = cast("Self", numpy.pad(self.value, pad_width, **kwargs).view(type(self)))
        # numpy.pad has stripped all metadata, so copy it over
        new.__metadata_finalize__(self)
        new._unit = self.unit
        # finally move the starting index based on the amount of left-padding
        new.x0 = new.x0 - self.dx * pad_width[0]
        return new

    def inject(
        self,
        other: Series,
        *,
        inplace: bool = False,
    ) -> Self:
        """Add two compatible `Series` along their shared x-axis values.

        Parameters
        ----------
        other : `Series`
            A `Series` whose xindex intersects with `self.xindex`.

        inplace : `bool`, optional
            If `True` (default) perform the operation in-place,
            modifying the current series.
            If `False` copy the data to new memory before modifying.

        Returns
        -------
        out : `Series`
            The sum of `self` and `other` along their shared x-axis values.

        Raises
        ------
        ValueError
            If `self` and `other` have incompatible units or xindex intervals.

        Notes
        -----
        The offset between ``self`` and ``other`` will be rounded to the nearest
        sample if they are not exactly aligned.

        If `self.xindex` is an array of timestamps, and if `other.xspan` is
        not a subset of `self.xspan`, then `other` will be cropped before
        being adding to `self`.

        Users may wish to taper or window their `Series` before
        passing it to this method. See :meth:`TimeSeries.taper` and
        :func:`~gwpy.signal.window.planck` for more information.
        """
        # check Series compatibility
        self.is_compatible(other)

        # crop to fit
        if (self.xunit == second) and (other.xspan[0] < self.xspan[0]):
            other = other.crop(start=self.xspan[0])
        if (self.xunit == second) and (other.xspan[1] > self.xspan[1]):
            other = other.crop(end=self.xspan[1])

        # find index offset
        ox0 = other.x0.to(self.x0.unit)
        idx = ((ox0 - self.x0) / self.dx).value
        if not idx.is_integer():
            warn(
                "Series have overlapping xspan but their x-axis values are not "
                "offset by an integer number of samples, "
                "will round to the nearest sample.",
                stacklevel=2,
            )
            idx = round(idx)

        # add the Series along their shared samples
        slice_ = slice(int(idx), int(idx) + other.shape[0])
        if inplace:
            out = self
        else:
            out = self.copy()
        out.value[slice_] += other.value
        return out
