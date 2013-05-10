#############################
`Segments` and `SegmentLists`
#############################

The core of the `gwpy.segments` module are the classes representing `Segment` objects.

=========
`Segment`
=========

The `Segment` object represents a single `[start ... stop)` GPS time pair::

    >>> from gwpy.segments import Segment
    >>> a = Segment(0, 10)

A single GPS time can be tested against the `Segment` as follows::

    >>> 5 in a
    True

Multiple `Segments` can be combined in a number of ways, including

    - union::

        >>> b = Segment(5, 15)
        >>> a & b
        Segment(5, 10)

    - intersection::

        >>> a + b
        Segment(0, 15)

    - difference::

        >>> a - b
        Segment(0, 5)

Boolean (`True`/`False`) tests can be performed in a number of ways:

    - Do two segments overlap?::

        >>> a.intersects(b)
        True

    - Are two segments completely non-overlapping? This method returns
      a positive number if the first segment covers an interval after
      the second, a negative number if the first segments covers an
      interval before the second, and 0 if the segments overlap or touch::

        >>> a.disjoint(b)
        0
        >>> c = Segment(15, 25)
        >>> a.disjoint(c)
        -1
        >>> c.disjoint(a)
        1

=============
`SegmentList`
=============

A number of `Segments` can be combined into a list, with which similar operations can be performed::

    >>> from gwpy.segments import (Segment, SegmentList)
    >>> a = Segment(0, 10)
    >>> b = Segment(20, 30)
    >>> l = SegmentList([a, b])
    >>> print(l)
    [Segment(0, 10), Segment(20, 30)]

Similar combinations can be made from `SegmentLists` as from `Segments`::

    >>> x = SegmentList([Segment(-10, 10)])
    >>> x |= SegmentList([Segment(20, 30)])
    >>> x -= SegmentList([Segment(-5, 5)])
    >>> print x
    [Segment(-10, -5), Segment(5, 10), Segment(20, 30)]

=================
`SegmentListDict`
=================

The `SegmentListDict` object provides a method to generate and record named sets of `SegmentLists`, extending the built-in python `dict`::

    >>> x = SegmentListDict()
    >>> x["H1"] = segmentlist([segment(0, 10)])
    >>> print x
    {'H1': [segment(0, 10)]}

In particular, methods for taking unions and intersections of the lists in the dictionary are available::

    >>> x["H2"] = segmentlist([segment(5, 15)])
    >>> x.intersection(["H1", "H2"])
    [segment(5, 10.0)]
