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

    # slice using a slice object
    if isinstance(slice_, slice):
        # try using index
        try:
            # new.xindex = old._yindex[slice]
            setattr(new, index(newaxis),
                    getattr(old, '_{}'.format(index(oldaxis)))[slice_])
        # no index, just set origin and delta
        except AttributeError:
            # new.x0 = old.y0 + (slice.start or 0) * old.dy
            setattr(new, origin(newaxis),
                    getattr(old, origin(oldaxis)) +
                    (slice_.start or 0) * getattr(old, delta(oldaxis)))
            # new.dx = old.dy * (slice.step or 1)
            setattr(new, delta(newaxis),
                    getattr(old, delta(oldaxis)) * (slice_.step or 1))

    # slice with an index array (always requires old.{x,y}index)
    else:
        setattr(new, index(newaxis), getattr(old, index(oldaxis))[slice_])

    return new
