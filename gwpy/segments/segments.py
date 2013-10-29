# Copyright (C) Duncan Macleod (2013)
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

from math import (ceil, floor)

from ..version import version as __version__
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Kipp Cannon <kipp.cannon@ligo.org>"

from glue.segments import (segment as _Segment,
                           segmentlist as _SegmentList,
                           segmentlistdict as _SegmentListDict)

from astropy.io import registry as io_registry

def _update_docstring(doc):
    """Update the docstring of the given class from glue to match
    the new naming for GWpy matching PEP-8.

    Parameters
    ----------
    doc : `str`
        input docstring to modify

    Returns
    -------
    newdoc : `str`
        the updated docstring with all references to the
        glue segments.segment* classes modified for GWpy with sphinx
        scope
    """
    doc = doc.replace('segmentlistdicts', '`SegmentListDicts <SegmentListDict>`')
    doc = doc.replace('segmentlistdict', '`SegmentListDict`')
    doc = doc.replace('Segmentlists', '`SegmentLists <SegmentList>`')
    doc = doc.replace('segmentlist', '`SegmentList`')
    doc = doc.replace('segments', '`Segments <Segment>`')
    doc = doc.replace('segment', '`Segment`')
    return doc


class Segment(_Segment):
    __doc__ = _update_docstring(_Segment.__doc__)
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
    __doc__ = _SegmentList.__doc__.replace('segmentlist', '`SegmentList`')
    def __repr__(self):
        return "<SegmentList([%s])>" % "\n              ".join(map(repr, self))

    def coalesce(self):
        self = super(SegmentList, self).coalesce()
        self = self.__class__([Segment(s[0], s[1]) for s in self])
        return self
    coalesce.__doc__ = _SegmentList.coalesce.__doc__

    def __str__(self):
        return "[%s]" % "\n ".join(map(str, self))

    read = classmethod(io_registry.read)


class SegmentListDict(_SegmentListDict):
    __doc__ = _update_docstring(_SegmentListDict.__doc__)

del _Segment
del _SegmentList
del _SegmentListDict
