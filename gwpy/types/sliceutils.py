# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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

from numbers import Integral

import numpy

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def format_nd_slice(item, ndim):
    """Preformat a getitem argument as an N-tuple
    """
    if not isinstance(item, tuple):
        item = (item,)
    return item[:ndim] + (None,) * (ndim - len(item))


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

    See also
    --------
    Series.__getitem__
    Array2D.__getitem__
    """
    slice_ = as_slice(slice_)

    # attribute names
    index = '{}index'.format
    origin = '{}0'.format
    delta = 'd{}'.format

    # if array has an index set already, use it
    if hasattr(old, '_{}index'.format(oldaxis)):
        setattr(new, index(newaxis), getattr(old, index(oldaxis))[slice_])

    # otherwise if using a slice, use origin and delta properties
    elif isinstance(slice_, slice) or not numpy.sum(slice_):
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


def null_slice(slice_):
    """Returns True if a slice will have no affect
    """
    try:
        slice_ = as_slice(slice_)
    except TypeError:
        return False

    if isinstance(slice_, numpy.ndarray) and numpy.all(slice_):
        return True
    if isinstance(slice_, slice) and slice_ in (
            slice(None, None, None), slice(0, None, 1)
    ):
        return True


def as_slice(slice_):
    """Convert an object to a slice, if possible
    """
    if isinstance(slice_, (Integral, numpy.integer, type(None))):
        return slice(0, None, 1)

    if isinstance(slice_, (slice, numpy.ndarray)):
        return slice_

    if isinstance(slice_, (list, tuple)):
        return tuple(map(as_slice, slice_))

    raise TypeError("Cannot format {!r} as slice".format(slice_))
