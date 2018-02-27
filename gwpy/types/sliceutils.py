# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018)
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

"""Utilities for the core types

These methods are designed for internal use only.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def format_nd_slice(item, ndim):
    """Preformat a getitem argument as an N-tuple
    """
    if not isinstance(item, tuple):
        item = (item,)
    if len(item) == ndim:
        return item
    return item + (None,) * (ndim - len(item))


def slice_axis_attributes(old, oldaxis, new, newaxis, slice_):
    """Set axis metadata for ``new`` by slicing an axis of ``old``

    This is primarily for internal use in slice functions (__getitem__)

    Parameters
    ----------
    old : `Array`
        array being sliced

    oldaxis : ``'x'`` or ``'y'``
        the axis to slice

    new : `Array`
        product of slice

    newaxis : ``'x'`` or ``'y'``
        the target axis

    slice_ : `slice`, `numpy.ndarray`
        the slice to apply to old (or an index array)

    See Also
    --------
    Series.__getitem__
    Array2D.__getitem__
    """
    # attribute names
    index = '{}index'.format
    origin = '{}0'.format
    delta = 'd{}'.format

    if isinstance(slice_, (int, type(None))):
        slice_ = slice(0, None, 1)

    # if array has an index set already, use it
    if hasattr(old, '_{}index'.format(oldaxis)):
        setattr(new, index(newaxis), getattr(old, index(oldaxis))[slice_])

    # otherwise if using a slice, use origin and delta properties
    elif isinstance(slice_, slice):
        offset = slice_.start or 0
        step = slice_.step or 1

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



    if len(will_slice) == 1:
        return will_slice[0]
    return


def null_slice(slice_):
    """Returns True if a slice will have no affect
    """
    if isinstance(slice_, (int, type(None))):
        return True

    if not isinstance(slice_, slice):
        return False

    if slice_ in (slice(None, None, None), slice(0, None, 1)):
        return True
