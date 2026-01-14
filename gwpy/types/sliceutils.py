# Copyright (c) 2018-2025 Cardiff University
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

"""Utilities for the core types.

These methods are designed for internal use only.
"""

from __future__ import annotations

from numbers import Integral
from typing import (
    TYPE_CHECKING,
    overload,
)

import numpy

if TYPE_CHECKING:
    from typing import (
        Literal,
        TypeAlias,
    )

    from numpy.typing import ArrayLike

    from . import Array

    SliceLike: TypeAlias = int | bool | list | slice | ArrayLike

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def format_nd_slice(
    item: SliceLike | tuple[SliceLike, ...],
    ndim: int,
) -> tuple[SliceLike | None, ...]:
    """Preformat a getitem argument as an N-tuple.

    Parameters
    ----------
    item : `object`, or a `tuple` of `object` instances
        The item to format as a tuple. If a tuple, it must be of length
        less than or equal to ``ndim``. If an object, it will be converted
        to a tuple of length ``ndim`` with the object as the first element.

    ndim : `int`
        The number of dimensions to format the item for.

    Returns
    -------
    item : `tuple`
        A tuple of length ``ndim`` with the item as the first element,
        and padded with `None` values if necessary.
    """
    if not isinstance(item, tuple):
        item = (item,)
    # pad the tuple with `None` until it's ``ndim`` elements
    return item[:ndim] + (None,) * (ndim - len(item))


def slice_axis_attributes(
    old: Array,
    oldaxis: Literal["x", "y", "z"],
    new: Array,
    newaxis: Literal["x", "y", "z"],
    slice_: SliceLike,
) -> Array:
    """Set axis metadata for ``new`` by slicing an axis of ``old``.

    This is primarily for internal use in slice functions (__getitem__)

    Parameters
    ----------
    old : `Array`
        Array being sliced.

    oldaxis : ``'x'`` or ``'y'``
        The axis to slice.

    new : `Array`
        Product of slice.

    newaxis : ``'x'`` or ``'y'``
        The target axis.

    slice_ : `slice`, `numpy.ndarray`
        The slice to apply to old (or an index array).

    See Also
    --------
    Series.__getitem__
    Array2D.__getitem__
    """
    slice_ = as_slice(slice_)

    # attribute names
    index = "{}index".format
    origin = "{}0".format
    delta = "d{}".format

    # if array has an index set already, use it
    if hasattr(old, f"_{oldaxis}index"):
        setattr(
            new,
            index(newaxis),
            getattr(old, index(oldaxis))[slice_],
        )
        return new

    # otherwise if using a slice, use origin and delta properties
    if isinstance(slice_, slice) or not numpy.sum(slice_):
        if isinstance(slice_, slice):
            offset = slice_.start or 0
            step = slice_.step or 1
        else:  # empty ndarray slice (so just set attributes)
            offset = 0
            step = 1

        dx = getattr(old, delta(oldaxis))
        x0 = getattr(old, origin(oldaxis))

        # set new.x0 / new.y0
        setattr(new, origin(newaxis), x0 + offset * dx)

        # set new.dx / new.dy
        setattr(new, delta(newaxis), dx * step)

    # otherwise slice with an index array
    else:
        setattr(new, index(newaxis), getattr(old, index(oldaxis))[slice_])

    return new


def null_slice(slice_: SliceLike) -> bool:
    """Return `True` if a slice will have no affect."""
    try:
        slice_ = as_slice(slice_)
    except TypeError:
        return False

    # trivial array slice
    if (
        isinstance(slice_, numpy.ndarray)
        and slice_.dtype == bool
        and slice_.all()
    ):
        return True

    # trivial slice object
    return (
        isinstance(slice_, slice)
        and slice_ in (
            slice(None, None, None),
            slice(0, None, 1),
        )
    )


@overload
def as_slice(slice_: SliceLike) -> slice | numpy.ndarray:
    ...


@overload
def as_slice(slice_: tuple[SliceLike, ...]) -> tuple[slice | numpy.ndarray, ...]:
    ...


def as_slice(
    slice_: None | SliceLike | tuple[SliceLike, ...],
) -> slice | numpy.ndarray | tuple[slice | numpy.ndarray, ...]:
    """Convert an object to a slice, or tuple of slices, if possible.

    slice_ : `int`, `list` `numpy.ndarray`, `None`
        The input to cast to a `slice_`.

    Returns
    -------
    slice : `slice` or `tuple` of `slice`
        The input cast into a single slice, or a tuple of slices.
    """
    if slice_ is None or isinstance(slice_, Integral | numpy.integer):
        return slice(0, None, 1)

    if isinstance(slice_, list):
        slice_ = numpy.array(slice_)

    if isinstance(slice_, slice | numpy.ndarray):
        return slice_

    if isinstance(slice_, tuple):
        return tuple(map(as_slice, slice_))

    msg = f"cannot format '{slice_}' as slice"
    raise TypeError(msg)
