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

"""A `Segment` is a interval of time marked by a GPS [start, stop)
semi-open interval. These typically represent periods when a
gravitational-wave laser interferometer was operating in a specific
configuration.
"""

from astropy.io import registry as io_registry

from ligo.segments import (segment, segmentlist, segmentlistdict)

from ..io.mp import read_multi as io_read_multi
from ..utils.decorators import return_as

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__all__ = ['Segment', 'SegmentList', 'SegmentListDict']


class Segment(segment):
    """A tuple defining a semi-open interval ``[start, end)``

    Each `Segment` represents the range of values in a given interval, with
    general arithmetic supported for combining/comparing overlapping segments.

    Parameters
    ----------
    start : `float`
        the start value of this `Segment`

    end : `float`
        the end value of this `Segment`

    Examples
    --------
    >>> Segment(0, 10) & Segment(5, 15)
    Segment(5, 10)
    >>> Segment(0, 10) | Segment(5, 15)
    Segment(0, 15)
    >>> Segment(0, 10) - Segment(5, 15)
    Segment(0, 5)
    >>> Segment(0, 10) < Segment(5, 15)
    True
    >>> Segment(1, 2) in Segment(0, 10)
    True
    >>> Segment(1, 11) in Segment(0, 10)
    False
    >>> Segment(0, 1)
    Segment(0, 1)
    >>> Segment(1, 0)
    Segment(0, 1)
    >>> bool(Segment(0, 1))
    True
    """
    @property
    def start(self):
        """The GPS start time of this segment
        """
        return self[0]

    @property
    def end(self):
        """The GPS end time of this segment
        """
        return self[1]

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self[0], self[1])

    def __str__(self):
        return "[%s ... %s)" % (self[0], self[1])


class SegmentList(segmentlist):
    """A `list` of `Segments <Segment>`

    The `SegmentList` provides additional methods that assist in the
    manipulation of lists of `Segments <Segment>`. In particular,
    arithmetic operations such as union and intersection are provided.
    Unlike the `Segment`, the `SegmentList` is closed under all supported
    arithmetic operations.

    All standard Python sequence-like operations are supported, like
    slicing, iteration and so on, but the arithmetic and other methods
    in this class generally expect the `SegmentList` to be in what is
    refered to as a "coalesced" state - consisting solely of disjoint
    `Segments <Segment>` listed in ascending order. Using the standard Python
    sequence-like operations, a `SegmentList` can be easily constructed
    that is not in this state;  for example by simply appending a
    `Segment` to the end of the list that overlaps some other `Segment`
    already in the list. The class provides a :meth:`~SegmentList.coalesce`
    method that can be called to put it in the coalesced state. Following
    application of the coalesce method, all arithmetic operations will
    function reliably. All arithmetic methods themselves return
    coalesced results, so there is never a need to call the coalesce
    method when manipulating a `SegmentList` exclusively via the
    arithmetic operators.

    Examples
    --------
    >>> x = SegmentList([Segment(-10, 10)])
    >>> x |= SegmentList([Segment(20, 30)])
    >>> x -= SegmentList([Segment(-5, 5)])
    >>> print(x)
    [Segment(-10, -5), Segment(5, 10), Segment(20, 30)]
    >>> print(~x)
    [Segment(-infinity, -10), Segment(-5, 5), Segment(10, 20),
     Segment(30, infinity)]
    """

    # -- representations ------------------------

    def __repr__(self):
        return "<SegmentList([%s])>" % "\n              ".join(map(repr, self))

    def __str__(self):
        return "[%s]" % "\n ".join(map(str, self))

    # -- type casting ---------------------------

    extent = return_as(Segment)(segmentlist.extent)

    def coalesce(self):
        super().coalesce()
        for i, seg in enumerate(self):
            self[i] = Segment(seg[0], seg[1])
        return self
    coalesce.__doc__ = segmentlist.coalesce.__doc__

    def to_table(self):
        """Convert this `SegmentList` to a `~astropy.table.Table`

        The resulting `Table` has four columns: `index`, `start`, `end`, and
        `duration`, corresponding to the zero-counted list index, GPS start
        and end times, and total duration in seconds, respectively.

        This method exists mainly to provide a way to write `SegmentList`
        objects in comma-separated value (CSV) format, via the
        :meth:`~astropy.table.Table.write` method.
        """
        from astropy.table import Table
        return Table(
            rows=[(i, s[0], s[1], abs(s)) for i, s in enumerate(self)],
            names=('index', 'start', 'end', 'duration'),
        )

    # -- i/o ------------------------------------

    @classmethod
    def read(cls, source, format=None, coalesce=False, **kwargs):
        # pylint: disable=redefined-builtin
        """Read segments from file into a `SegmentList`

        Parameters
        ----------
        filename : `str`
            path of file to read

        format : `str`, optional
            source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        coalesce : `bool`, optional
            if `True` coalesce the segment list before returning,
            otherwise return exactly as contained in file(s).

        **kwargs
            other keyword arguments depend on the format, see the online
            documentation for details (:ref:`gwpy-segments-io`)

        Returns
        -------
        segmentlist : `SegmentList`
            `SegmentList` active and known segments read from file.

        Raises
        ------
        IndexError
            if ``source`` is an empty list

        Notes
        -----"""
        def combiner(listofseglists):
            """Combine `SegmentList` from each file into a single object
            """
            out = cls(seg for seglist in listofseglists for seg in seglist)
            if coalesce:
                return out.coalesce()
            return out

        return io_read_multi(combiner, cls, source, format=format, **kwargs)

    def write(self, target, *args, **kwargs):
        """Write this `SegmentList` to a file

        Arguments and keywords depend on the output format, see the
        online documentation for full details for each format.

        Parameters
        ----------
        target : `str`
            output filename

        Notes
        -----"""
        return io_registry.write(self, target, *args, **kwargs)


class SegmentListDict(segmentlistdict):
    """A `dict` of `SegmentLists <SegmentList>`

    This class implements a standard mapping interface, with additional
    features added to assist with the manipulation of a collection of
    `SegmentList` objects. In particular, methods for taking unions and
    intersections of the lists in the dictionary are available, as well
    as the ability to record and apply numeric offsets to the
    boundaries of the `Segments <Segment>` in each list.

    The numeric offsets are stored in the "offsets" attribute, which
    itself is a dictionary, associating a number with each key in the
    main dictionary. Assigning to one of the entries of the offsets
    attribute has the effect of shifting the corresponding `SegmentList`
    from its original position (not its current position) by the given
    amount.

    Examples
    --------
    >>> x = SegmentListDict()
    >>> x["H1"] = SegmentList([Segment(0, 10)])
    >>> print(x)
    {'H1': [Segment(0, 10)]}
    >>> x.offsets["H1"] = 6
    >>> print(x)
    {'H1': [Segment(6.0, 16.0)]}
    >>> x.offsets.clear()
    >>> print(x)
    {'H1': [Segment(0.0, 10.0)]}
    >>> x["H2"] = SegmentList([Segment(5, 15)])
    >>> x.intersection(["H1", "H2"])
    [Segment(5, 10.0)]
    >>> x.offsets["H1"] = 6
    >>> x.intersection(["H1", "H2"])
    [Segment(6.0, 15)]
    >>> c = x.extract_common(["H1", "H2"])
    >>> c.offsets.clear()
    >>> c
    {'H2': [Segment(6.0, 15)], 'H1': [Segment(0.0, 9.0)]}
    """
    pass


# clean up the namespace
del segment, segmentlist, segmentlistdict
