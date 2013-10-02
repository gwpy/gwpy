# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""A `Segment` is a interval of time marked by a GPS [start, stop)
semi-open interval. These typically represent periods when a
gravitational-wave laser interferometer was operating in a specific
configuration.
"""

from math import (ceil, floor)

from ..version import version as __version__
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Kipp Cannon <kipp.cannon@ligo.org>"

from glue.segments import (segment as _Segment,
                           segmentlist as _SegmentList,
                           segmentlistdict as SegmentListDict)


class Segment(_Segment):
    """A [GPS start, GPS end) semi-open interval, representing
    a period of time.
    """
    @property
    def start(self):
        return self[0]

    @property
    def end(self):
        return self[1]

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self[0], self[1])

    def __str__(self):
        return "[%s ... %s)" % (self[0], self[1])

class SegmentList(_SegmentList):
    def __repr__(self):
        return "<SegmentList([%s])>" % "\n              ".join(map(repr, self))

    def coalesce(self):
        self = super(SegmentList, self).coalesce()
        self = self.__class__([Segment(s[0], s[1]) for s in self])
        return self
    coalesce.__doc__ = _SegmentList.coalesce.__doc__

    def __str__(self):
        return "[%s]" % "\n ".join(map(str, self))

del _Segment
del _SegmentList
