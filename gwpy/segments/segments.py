# -*- coding: utf-8 -*-
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

from glue.segments import (segment as _Segment,
                           segmentlist as _SegmentList,
                           segmentlistdict as _SegmentListDict)

from astropy.io import registry as io_registry

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Kipp Cannon <kipp.cannon@ligo.org>"
__all__ = ['Segment', 'SegmentList', 'SegmentListDict']


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
    doc = doc.replace('segmentlistdicts',
                      '`SegmentListDicts <SegmentListDict>`')
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


class SegmentList(_SegmentList):
    __doc__ = _update_docstring(_SegmentList.__doc__)

    def __repr__(self):
        return "<SegmentList([%s])>" % "\n              ".join(map(repr, self))

    def coalesce(self):
        self = super(SegmentList, self).coalesce()
        for i, s in enumerate(self):
            self[i] = Segment(s[0], s[1])
        return self
    coalesce.__doc__ = _SegmentList.coalesce.__doc__

    def __str__(self):
        return "[%s]" % "\n ".join(map(str, self))

    # -- i/o ------------------------------------

    @classmethod
    def read(cls, source, format=None, **kwargs):
        """Read segments from file into a `SegmentList`

        Parameters
        ----------
        filename : `str`
            path of file to read

        format : `str`, optional
            source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        Returns
        -------
        segmentlist : `SegmentList`
            `SegmentList` active and known segments read from file.

        Notes
        -----"""
        return io_registry.read(cls, source, format=format, **kwargs)

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


class SegmentListDict(_SegmentListDict):
    __doc__ = _update_docstring(_SegmentListDict.__doc__)
