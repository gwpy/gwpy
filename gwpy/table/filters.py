# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Filter functions for use with :meth:`EventTable.filter`

Each of these functions has the same input and output format.
The inputs must be

- ``columns`` (`str`) : the name of the column to use
- ``operand`` : the operand to the filter

No other arguments or keywords are supported.

The output must be a index array indicating for each row in the
associated table whether it should be returned by the filter or not:
`True` means to be returned, and `False` to be discarded.
"""

import numpy


def in_segmentlist(column, segmentlist):
    """Return the index of values lying inside the given segmentlist

    A `~gwpy.segments.Segment` represents a semi-open interval,
    so for any segment `[a, b)`, a value `x` is 'in' the segment if

    a <= x < b
    """
    segmentlist = type(segmentlist)(segmentlist).coalesce()
    idx = column.argsort()
    contains = numpy.zeros(column.shape[0], dtype=bool)
    j = 0
    try:
        segstart, segend = segmentlist[j]
    except IndexError:  # no segments, return all False
        return contains
    i = 0
    while i < contains.shape[0]:
        # extract time for this index
        x = idx[i]  # <- index in original column
        time = column[x]
        # if before start, move to next value
        if time < segstart:
            i += 1
            continue
        # if after end, find the next segment and check value again
        if time >= segend:
            j += 1
            try:
                segstart, segend = segmentlist[j]
                continue
            except IndexError:
                break
        # otherwise value must be in this segment
        contains[x] = True
        i += 1
    return contains


def not_in_segmentlist(column, segmentlist):
    """Return the index of values not lying inside the given segmentlist

    See :func:`~gwpy.table.filters.in_segmentlist` for more details
    """
    return in_segmentlist(column, ~segmentlist)
