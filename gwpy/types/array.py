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

"""The `Array`.

The `Array` structure provides the core array-with-metadata environment
with the standard array methods wrapped to return instances of itself.

Each sub-class of `Array` should override the `Array._metadata_slots`
attribute, giving a list of the valid properties for these data. This is
critical to being able to view data with this class, used when copying and
transforming instances of the class.
"""

from __future__ import annotations

import contextlib
import copy
import os
import textwrap
from decimal import Decimal
from math import modf
from typing import TYPE_CHECKING

import numpy
from astropy.units import Quantity
from astropy.utils.compat.numpycompat import COPY_IF_NEEDED

from ..detector import Channel
from ..detector.units import parse_unit
from ..time import Time, to_gps

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )
    from typing import (
        ClassVar,
        Literal,
        Self,
    )

    from astropy.units import UnitBase
    from astropy.units.typing import QuantityLike
    from numpy.typing import DTypeLike

    from ..time import SupportsToGps
    from ..typing import UnitLike
    from .sliceutils import SliceLike

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

numpy.set_printoptions(threshold=200, linewidth=65)


# -- core Array ----------------------

class Array(Quantity):
    """Array holding data with a unit, and other metadata.

    This `Array` holds the input data and a standard set of metadata
    properties associated with GW data.

    Parameters
    ----------
    value : array-like
        Input data array.

    unit : `~astropy.units.Unit`
        Physical unit of these data.

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine

    name : `str`
        Descriptive title for this array.

    channel : `~gwpy.detector.Channel`, `str`
        Source data stream for these data.

    dtype : `~numpy.dtype`
        Input data type.

    copy : `bool`
        Choose to copy the input data to new memory.

    subok : `bool`
        Allow passing of sub-classes by the array generator.

    Returns
    -------
    array : `Array`
        A new array, with a view of the data, and all associated metadata.

    Examples
    --------
    To create a new `Array` from a list of samples:

    >>> a = Array([1, 2, 3, 4, 5], 'm/s', name='my data')
    >>> print(a)
    Array([ 1., 2., 3., 4., 5.]
          unit: Unit("m / s"),
          name: 'my data',
          epoch: None,
          channel: None)
    """

    #: list of new attributes defined in this class
    #
    # this is used in __array_finalize__ to create new instances of this
    # object [http://docs.scipy.org/doc/numpy/user/basics.subclassing.html]
    _metadata_slots: ClassVar[tuple[str, ...]] = (
        "name",
        "epoch",
        "channel",
    )

    def __new__(
        cls,
        value: QuantityLike,
        *,
        # Quantity attrs
        unit: UnitBase | str | None = None,
        # new attrs
        name: str | None = None,
        epoch: SupportsToGps | None = None,
        channel: Channel | str | None = None,
        # ndarray attrs
        dtype: DTypeLike = None,
        copy: bool = True,
        subok: bool = True,
        order: str | None = None,
        ndmin: int = 0,
    ) -> Self:
        """Create a new `Array`."""
        # pick dtype from input array
        if dtype is None and isinstance(value, numpy.ndarray):
            dtype = value.dtype

        # parse unit with forgiveness
        if unit is not None:
            unit = parse_unit(unit, parse_strict="warn")

        # create new array
        new = super().__new__(
            cls,
            value,
            unit=unit,
            dtype=dtype,
            copy=COPY_IF_NEEDED,
            order=order,
            subok=subok,
            ndmin=ndmin,
        )

        # explicitly copy here to get ownership of the data,
        # see (astropy/astropy#7244)
        if copy:
            new = new.copy()

        # set new attributes
        if name is not None:
            new.name = name
        if epoch is not None:
            new.epoch = epoch
        if channel is not None:
            new.channel = channel

        return new

    # -- object creation -------------
    # methods here handle how these objects are created,
    # mainly to do with making sure metadata attributes get
    # properly reassigned from old to new

    def _wrap_function(
        self,
        function: Callable,
        *args,  # noqa: ANN002
        **kwargs,
    ) -> Self | Quantity:
        # if the output of the function is a scalar, return it as a Quantity
        # not whatever class this is
        out = super()._wrap_function(function, *args, **kwargs)
        if out.ndim == 0:
            return Quantity(out.value, out.unit)
        return out

    def __quantity_subclass__(
        self,
        unit: UnitBase,
    ) -> tuple[type, Literal[True]]:
        """View operations should return the same type (or a subclass)."""
        return type(self), True

    def __array_finalize__(self, obj: Self | None) -> None:
        """Finalise this `Array`.

        This is called whenever a new view of an `Array` is created.
        """
        # format a new instance of this class starting from `obj`
        if obj is None:
            return

        # call Quantity.__array_finalize__ to handle the units
        super().__array_finalize__(obj)

        # then update metadata
        if isinstance(obj, Quantity):
            self.__metadata_finalize__(obj, force=False)

    def __metadata_finalize__(
        self,
        obj: Self,
        *,
        force: bool = False,
    ) -> None:
        """Finalise metadata for this `Array`."""
        # apply metadata from obj to self if creating a new object
        for attr in self._metadata_slots:
            _attr = f"_{attr}"  # use private attribute (not property)
            # if attribute is unset, default it to None, then update
            # from obj if desired
            try:
                getattr(self, _attr)
            except AttributeError:
                update = True
            else:
                update = force
            if update:
                try:
                    val = getattr(obj, _attr)
                except AttributeError:
                    continue
                else:
                    if isinstance(val, Quantity):  # copy Quantities
                        setattr(self, _attr, type(val)(val))
                    else:
                        setattr(self, _attr, val)

    def __getitem__(
        self,
        item: SliceLike | tuple[SliceLike, ...],
    ) -> Self | Quantity:
        """Get an item from this `Array`.

        This implements ``array[item]``.
        """
        new = super().__getitem__(item)

        # return scalar as a Quantity
        if numpy.ndim(new) == 0:
            return Quantity(new, unit=self.unit)

        return new

    # -- display ---------------------

    def _repr_helper(self, print_: Callable) -> str:
        """Create a string representation of an `Array`."""
        if print_ is repr:
            opstr = "="
        else:
            opstr = ": "

        # get prefix and suffix
        prefix = f"{type(self).__name__}("
        suffix = ")"
        if print_ is repr:
            prefix = f"<{prefix}"
            suffix += ">"

        indent = " " * len(prefix)

        # format value
        arrstr = numpy.array2string(
            self.value,
            separator=", ",
            prefix=prefix,
        )

        # format unit
        metadata = [("unit", print_(self.unit) or "dimensionless")]

        # format other metadata
        try:
            attrs = self._print_slots
        except AttributeError:
            attrs = self._metadata_slots
        for attr in attrs:
            try:
                val = getattr(self, attr)
            except (AttributeError, KeyError):
                val = None
            thisindent = indent + " " * (len(attr) + len(opstr))
            metadata.append((
                attr.lstrip("_"),
                textwrap.indent(print_(val), thisindent).strip(),
            ))
        metadatastr = f",{os.linesep}{indent}".join(
            f"{attr}{opstr}{value}" for attr, value in metadata
        )

        return f"{prefix}{arrstr},{os.linesep}{indent}{metadatastr}{suffix}"

    def __repr__(self) -> str:
        """Return a representation of this object.

        This just represents each of the metadata objects appropriately
        after the core data array
        """
        return self._repr_helper(repr)

    def __str__(self) -> str:
        """Return a printable string format representation of this object.

        This just prints each of the metadata objects appropriately
        after the core data array
        """
        return self._repr_helper(str)

    # -- new properties --------------

    # name
    @property
    def name(self) -> str | None:
        """Name for this data set."""
        self._name: str | None
        try:
            return self._name
        except AttributeError:
            self._name = None
            return self._name

    @name.setter
    def name(self, val: str | None) -> None:
        if val is None:
            self._name = None
        else:
            self._name = str(val)

    @name.deleter
    def name(self) -> None:
        with contextlib.suppress(AttributeError):
            del self._name

    # epoch
    @property
    def epoch(self) -> Time | None:
        """GPS epoch associated with these data."""
        self._epoch: Time | None
        try:
            if self._epoch is None:
                return None
            return Time(*modf(self._epoch)[::-1], format="gps", scale="utc")
        except AttributeError:
            self._epoch = None
            return self._epoch

    @epoch.setter
    def epoch(self, epoch: SupportsToGps | None) -> None:
        if epoch is None:
            self._epoch = None
        else:
            self._epoch = Decimal(str(to_gps(epoch)))

    @epoch.deleter
    def epoch(self) -> None:
        with contextlib.suppress(AttributeError):
            del self._epoch

    # channel
    @property
    def channel(self) -> Channel | None:
        """Instrumental channel associated with these data."""
        self._channel: Channel | None
        try:
            return self._channel
        except AttributeError:
            self._channel = None
            return self._channel

    @channel.setter
    def channel(self, chan: Channel | str | None) -> None:
        if isinstance(chan, Channel):
            self._channel = chan
        elif chan is None:
            self._channel = None
        else:
            self._channel = Channel(chan)

    @channel.deleter
    def channel(self) -> None:
        with contextlib.suppress(AttributeError):
            del self._channel

    # unit - we override this to make the property less pedantic
    #        astropy won't allow you to set a unit that it doesn't
    #        recognise
    @property
    def unit(self) -> UnitBase | None:
        """The physical unit of these data."""
        try:
            return self._unit
        except AttributeError:
            return None

    @unit.setter
    def unit(self, unit: UnitBase | str | None) -> None:
        if getattr(self, "_unit", None) is not None:
            msg = (
                "can't set attribute; to change the units of this "
                f"{type(self).__name__}, use the .to() instance method "
                "instead, otherwise use the override_unit() instance method "
                "to forcefully set a new unit",
            )
            raise AttributeError(msg)
        self._unit = parse_unit(unit)

    @unit.deleter
    def unit(self) -> None:
        with contextlib.suppress(AttributeError):
            del self._unit

    # -- array methods ---------------

    def __array_ufunc__(
        self,
        function: Callable,
        method: str,
        *inputs,  # noqa: ANN002
        **kwargs,
    ) -> Self | Quantity:
        """Wrap a ufunc, handling units, and scalar outputs."""
        out = super().__array_ufunc__(function, method, *inputs, **kwargs)
        # if a ufunc returns a scalar, return a Quantity
        if not out.ndim:
            return Quantity(out, copy=COPY_IF_NEEDED)
        # otherwise return an array
        return out

    def abs(
        self,
        **kwargs,
    ) -> Self | Quantity:
        """Return the absolute value of the data in this `Array`.

        See Also
        --------
        numpy.absolute
            For details of all available positional and keyword arguments,
            and for details of the return value.
        """
        return self._wrap_function(
            numpy.absolute,
            **kwargs,
        )

    def median(
        self,
        axis: int | Iterable[int] | None = None,
        **kwargs,
    ) -> Self | Quantity:
        """Return the median of the data in this `Array`.

        See Also
        --------
        numpy.median
            For details of all available positional and keyword arguments,
            and for details of the return value.
        """
        return self._wrap_function(
            numpy.median,
            axis=axis,
            **kwargs,
        )

    def _to_own_unit(
        self,
        value: QuantityLike,
        *,
        check_precision: bool = True,
        unit: UnitBase | None = None,
    ) -> Self:
        if unit is None and self.unit is None:
            unit = ""
        return super()._to_own_unit(
            value,
            check_precision=check_precision,
            unit=unit,
        )

    _to_own_unit.__doc__ = Quantity._to_own_unit.__doc__  # noqa: SLF001

    def override_unit(
        self,
        unit: UnitLike,
        parse_strict: Literal["raise", "warn", "silent"] = "raise",
    ) -> None:
        """Reset the unit of these data.

        Use of this method is discouraged in favour of `to()`,
        which performs accurate conversions from one unit to another.
        The method should really only be used when the original unit of the
        array is plain wrong.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, `str`
            the unit to force onto this array

        parse_strict : `str`
            how to handle errors in the unit parsing, default is to
            raise the underlying exception from `astropy.units`

        See Also
        --------
        gwpy.detector.units.parse_unit
            For details of unit string parsing.
        """
        self._unit = parse_unit(unit, parse_strict=parse_strict)

    def flatten(self, order: str = "C") -> Quantity:
        """Return a copy of the array collapsed into one dimension.

        Any index information is removed as part of the flattening,
        and the result is returned as a `~astropy.units.Quantity` array.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}
            'C' means to flatten in row-major (C-style) order.
            'F' means to flatten in column-major (Fortran-
            style) order. 'A' means to flatten in column-major
            order if `a` is Fortran *contiguous* in memory,
            row-major order otherwise. 'K' means to flatten
            `a` in the order the elements occur in memory.
            The default is 'C'.

        Returns
        -------
        y : `~astropy.units.Quantity`
            A copy of the input array, flattened to one dimension.

        See Also
        --------
        ravel : Return a flattened array.
        flat : A 1-D flat iterator over the array.

        Examples
        --------
        >>> a = Array([[1,2], [3,4]], unit='m', name='Test')
        >>> a.flatten()
        <Quantity [1., 2., 3., 4.] m>
        """
        return super().flatten(order=order).view(Quantity)

    def copy(self, order: Literal["C", "F", "A", "K"] = "C") -> Self:
        """Return a copy of this `Array`.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            The desired memory layout order for the copy.

        See Also
        --------
        numpy.ndarray.copy
            For details of the copy operation.
        """
        out = super().copy(order=order)
        for slot in self._metadata_slots:
            old = getattr(self, f"_{slot}", None)
            if old is not None:
                setattr(out, slot, copy.copy(old))
        return out

    copy.__doc__ = Quantity.copy.__doc__

    # -- type helpers ----------------

    def __mul__(
        self,
        other: QuantityLike,
    ) -> Self:
        """Multiply this `Array` by another quantity."""
        return super().__mul__(other)

    def __pow__(
        self,
        power: QuantityLike,
    ) -> Self:
        """Raise this `Array` to a power."""
        return super().__pow__(power)
