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

"""`Array2D` - a two-dimensional, indexed array."""

from __future__ import annotations

import contextlib
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)
from warnings import warn

from astropy.units import (
    Quantity,
    Unit,
)

from ..io.registry import UnifiedReadWriteMethod
from . import sliceutils
from .connect import (
    Array2DRead,
    Array2DWrite,
)
from .index import Index
from .series import Series

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import (
        ClassVar,
        Self,
    )

    from astropy.units import UnitBase

    from ..plot import Plot
    from ..segments import Segment
    from ..typing import QuantityLike
    from .sliceutils import SliceLike

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class Array2D(Series):
    """A two-dimensional array with metadata.

    Parameters
    ----------
    value : array-like
        Input data array.

    unit : `~astropy.units.Unit`, optional
        Physical unit of these data.

    x0 : `float`, `~astropy.units.Quantity`, optional
        The starting value for the x-axis of this array.

    dx : `float`, `~astropy.units.Quantity`, optional
        The step size for the x-axis of this array.

    xindex : `array-like`
        The complete array of x-axis values for this array. This argument
        takes precedence over `x0` and `dx` so should be
        given in place of these if relevant, not alongside.

    xunit : `~astropy.units.Unit`, optional
        The unit of the x-axis coordinates. If not given explicitly, it will be
        taken from any of `dx`, `x0`, or `xindex`, or set to a boring default.

    y0 : `float`, `~astropy.units.Quantity`, optional
        The starting value for the y-axis of this array.

    dy : `float`, `~astropy.units.Quantity`, optional
        The step size for the y-axis of this array.

    yindex : `array-like`
        The complete array of y-axis values for this array. This argument
        takes precedence over `y0` and `dy` so should be
        given in place of these if relevant, not alongside.

    yunit : `~astropy.units.Unit`, optional
        The unit of the y-axis coordinates. If not given explicitly, it will be
        taken from any of `dy`, `y0`, or `yindex`, or set to a boring default.

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine.

    name : `str`, optional
        Descriptive title for this array.

    channel : `~gwpy.detector.Channel`, `str`, optional
        Source data stream for these data.

    dtype : `~numpy.dtype`, optional
        Input data type.

    copy : `bool`, optional
        Choose to copy the input data to new memory.

    subok : `bool`, optional
        Allow passing of sub-classes by the array generator.

    Returns
    -------
    array : `Array`
        a new array, with a view of the data, and all associated metadata
    """

    _metadata_slots: ClassVar[tuple[str, ...]] = (
        *Series._metadata_slots,
        "y0",
        "dy",
        "yindex",
    )

    #: The default unit for the X-axis index.
    _default_xunit: ClassVar[UnitBase] = Unit("")

    #: The default unit for the Y-axis index.
    _default_yunit: ClassVar[UnitBase] = Unit("")

    #: The class used for viewing a row of this array.
    _rowclass: ClassVar[type[Series]] = Series

    #: The class used for viewing a column of this array.
    _columnclass: ClassVar[type[Series]] = Series

    #: The number of dimensions of this array.
    _ndim: ClassVar[int] = 2

    def __new__(
        cls,
        data: QuantityLike,
        unit: UnitBase | str | None = None,
        x0: QuantityLike | None = None,
        dx: QuantityLike | None = None,
        xindex: QuantityLike | None = None,
        xunit: UnitBase | str | None = None,
        y0: QuantityLike | None = None,
        dy: QuantityLike | None = None,
        yindex: QuantityLike | None = None,
        yunit: UnitBase | str | None = None,
        **kwargs,
    ) -> Self:
        """Define a new `Array2D`."""
        # create new object
        new = super().__new__(
            cls,
            data,
            unit=unit,
            xindex=xindex,
            xunit=xunit,
            x0=x0,
            dx=dx,
            **kwargs,
        )

        # set y-axis metadata from yindex
        if yindex is not None:
            # warn about duplicate settings
            if dy is not None:
                warn(
                    f"yindex was given to {cls.__name__}(), dy will be ignored",
                    stacklevel=2,
                )
            if y0 is not None:
                warn(
                    f"yindex was given to {cls.__name__}(), y0 will be ignored",
                    stacklevel=2,
                )
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

    @overload
    def __getitem__(self, item: tuple[int, int]) -> Quantity: ...
    @overload
    def __getitem__(self, item: int) -> Series: ...
    @overload
    def __getitem__(self, item: SliceLike | tuple[SliceLike, ...]) -> Self: ...

    # rebuild getitem to handle complex slicing
    def __getitem__(  # ty: ignore[invalid-method-override]
        self,
        item: tuple[int, int] | int | SliceLike | tuple[SliceLike, ...],
    ) -> Self | Series | Quantity:
        """Get an item or a slice from this `Array2D."""
        new = super().__getitem__(item)

        # Slice axis 1 metadata
        colslice, rowslice = sliceutils.format_nd_slice(item, self.ndim)

        # Column slice
        if new.ndim == 1 and isinstance(colslice, int):
            new = cast("Series", new.view(self._columnclass))
            del new.xindex
            new.__metadata_finalize__(self)
            sliceutils.slice_axis_attributes(self, "y", new, "x", rowslice)

        # Row slice
        elif new.ndim == 1:
            new = cast("Series", new.view(self._rowclass))

        # Slice axis 1 for Array2D
        # (Series.__getitem__ will have performed column slice already)
        elif new.ndim > 1 and not sliceutils.null_slice(rowslice):
            sliceutils.slice_axis_attributes(self, "y", new, "y", rowslice)

        return new

    def __array_finalize__(self, obj: Self | None) -> None:
        """Finalize the array after a view is created."""
        super().__array_finalize__(obj)

        # Series.__array_finalize__ might set _yindex to None, so delete it
        if getattr(self, "_yindex", 0) is None:
            del self.yindex

    def __iter__(self) -> Quantity:
        """Yield the columns of this `Array2D` as `Quantity` objects."""
        # astropy Quantity.__iter__ does something fancy that we don't need
        # because we overload __getitem__
        return super(Quantity, self).__iter__()

    # -- Array2d properties ----------

    @property
    def y0(self) -> Quantity:
        """Y-axis coordinate of the first data point."""
        self._y0: Quantity
        try:
            return self._y0
        except AttributeError:
            try:
                self._y0 = self._yindex[0]
            except (AttributeError, IndexError):
                self._y0 = Quantity(0, self.yunit)
            return self._y0

    @y0.setter
    def y0(self, value: QuantityLike) -> None:
        self._update_index("y", "y0", value)

    @y0.deleter
    def y0(self) -> None:
        with contextlib.suppress(AttributeError):
            del self._y0

    @property
    def dy(self) -> Quantity:
        """Y-axis sample separation."""
        self._dy: Quantity
        try:
            return self._dy
        except AttributeError:
            try:
                self._yindex  # noqa: B018
            except AttributeError:
                self._dy = Quantity(1, self.yunit)
            else:
                if not self.yindex.regular:
                    msg = (
                        "this series has an irregular y-axis "
                        "index, so 'dy' is not well defined"
                    )
                    raise AttributeError(msg)
                self._dy = self.yindex[1] - self.yindex[0]
            return self._dy

    @dy.setter
    def dy(self, value: QuantityLike) -> None:
        self._update_index("y", "dy", value)

    @dy.deleter
    def dy(self) -> None:
        with contextlib.suppress(AttributeError):
            del self._dy

    @property
    def yunit(self) -> UnitBase:
        """Unit of Y-axis index."""
        try:
            return self._dy.unit
        except AttributeError:
            try:
                return self._y0.unit
            except AttributeError:
                return self._default_yunit

    @property
    def yindex(self) -> Index:
        """Positions of the data on the y-axis."""
        self._yindex: Index
        try:
            return self._yindex
        except AttributeError:
            self._yindex = Index.define(self.y0, self.dy, self.shape[1])
            return self._yindex

    @yindex.setter
    def yindex(self, index: QuantityLike) -> None:
        self._set_index("y", index)

    @yindex.deleter
    def yindex(self) -> None:
        with contextlib.suppress(AttributeError):
            del self._yindex

    @property
    def yspan(self) -> Segment:
        """Y-axis [low, high) segment encompassed by these data.

        :type: `~gwpy.segments.Segment`
        """
        return self._index_span("y")

    @property
    def T(self) -> Self:  # noqa: N802
        """Return the transpose of this `Array2D`."""
        trans = self.value.T.view(type(self))
        trans.__array_finalize__(self)
        if hasattr(self, "_xindex"):
            trans.yindex = self.xindex.view()
        else:
            trans.y0 = self.x0
            trans.dy = self.dx
        if hasattr(self, "_yindex"):
            trans.xindex = self.yindex.view()
        else:
            trans.x0 = self.y0
            trans.dx = self.dy
        return trans

    # -- Array2D i/o -----------------

    read = UnifiedReadWriteMethod(Array2DRead)
    write = UnifiedReadWriteMethod(Array2DWrite)

    # -- Array2D methods -------------

    def _check_compatible_gwpy(
        self,
        other: Quantity,
        *,
        irregular_equal: bool = True,
    ) -> None:
        """Check whether this `Array2D` and another are compatible.

        This method checks that the Index arrays are compatible.
        """
        self._check_compatible_index(
            other,
            axis="y",
            irregular_equal=irregular_equal,
        )
        super()._check_compatible_gwpy(
            other,
            irregular_equal=irregular_equal,
        )

    def value_at(  # type: ignore[override]
        self,
        x: QuantityLike,
        y: QuantityLike,
    ) -> Quantity:
        """Return the value of this `Series` at the given `(x, y)` coordinates.

        Parameters
        ----------
        x : `float`, `~astropy.units.Quantity`
            The `xindex` value at which to search.

        y : `float`, `~astropy.units.Quantity`
            The `yindex` value at which to search.

        Returns
        -------
        z : `~astropy.units.Quantity`
            The value of this Series at the given coordinates.

        Raises
        ------
        IndexError
            If ``x`` or `y`` don't match a value on their respective index.
        """
        x = Quantity(x, self.xindex.unit).value
        y = Quantity(y, self.yindex.unit).value
        try:
            idx = (self.xindex.value == x).nonzero()[0][0]
        except IndexError as exc:
            exc.args = (f"Value {x!r} not found in array xindex",)
            raise
        try:
            idy = (self.yindex.value == y).nonzero()[0][0]
        except IndexError as exc:
            exc.args = ("Value %r not found in array yindex",)
            raise
        return self[idx, idy]

    def imshow(self, **kwargs) -> Plot:
        """Render this array on a `Plot` using `~matplotlib.pyplot.imshow`.

        Parameters
        ----------
        kwargs
            All arguments are passed to `plot`.

        See Also
        --------
        plot
            For details of plotting this object.
        """
        return self.plot(method="imshow", **kwargs)

    def pcolormesh(self, **kwargs) -> Plot:
        """Render this array on a `Plot` using `~matplotlib.pyplot.pcolormesh`.

        Parameters
        ----------
        kwargs
            All arguments are passed to `plot`.

        See Also
        --------
        plot
            For details of plotting this object.
        """
        return self.plot(method="pcolormesh", **kwargs)

    def plot(self, method: str = "imshow", **kwargs) -> Plot:
        """"Render this array on a `Plot`.

        Parameters
        ----------
        method : `str`, optional
            The `~gwpy.plot.Axes` method to call when rendering this array.

        kwargs
            All arguments are passed to `plot`.

        See Also
        --------
        gwpy.plot.Plot
            For details of plotting this object.
        """
        from ..plot import Plot

        # correct for log scales and zeros
        if kwargs.get("xscale") == "log" and self.x0.value == 0:
            kwargs.setdefault("xlim", (self.dx.value, self.xspan[1]))
        if kwargs.get("yscale") == "log" and self.y0.value == 0:
            kwargs.setdefault("ylim", (self.dy.value, self.yspan[1]))

        # make plot
        return Plot(self, method=method, **kwargs)

    # -- Array2D modifiers -----------
    # all of these try to return Quantities rather than simple numbers

    def _wrap_function(self, function: Callable, *args, **kwargs) -> Self | Quantity:
        """Wrap a function to return a new `Array2D` or `Quantity`."""
        out = super()._wrap_function(function, *args, **kwargs)

        if out.ndim != 1:
            return out

        # -- return Series
        try:
            axis = kwargs["axis"]
        except KeyError:
            axis = args[0]
        metadata = {
            "unit": out.unit,
            "channel": out.channel,
            "epoch": self.epoch,
            "name": f"{self.name} {function.__name__}",
        }

        # return Column series
        if axis == 0:
            if hasattr(self, "_yindex"):
                metadata["xindex"] = self.yindex
            else:
                metadata["x0"] = self.y0
                metadata["dx"] = self.dy
            return self._columnclass(out.value, **metadata)

        # return Row series
        if hasattr(self, "_xindex"):
            metadata["xindex"] = self.xindex
        else:
            metadata["x0"] = self.x0
            metadata["dx"] = self.dx
        return self._rowclass(out.value, **metadata)
