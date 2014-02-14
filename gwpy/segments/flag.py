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

"""This module defines the `DataQualityFlag` object, representing a
named set of segments used in defining GW instrument state or
data quality information.
"""

import re
import warnings
from copy import copy as shallowcopy
from math import (floor, ceil)

try:
    from collections import OrderedDict
except ImportError:
    from astropy.utils import OrderedDict

from astropy.io import registry as io_registry

from .segments import Segment, SegmentList

from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = ['DataQualityFlag', 'DataQualityDict']


class DataQualityFlag(object):
    """A representation of a named set of segments.

    Parameters
    ----------
    name : str, optional
        The name of this data quality flag. This should be of the form
        IFO:DQ-FLAG_NAME:VERSION, e.g. 'H1:DMT-SCIENCE:1'
    active : `~segments.SegmentList`, optional
        A list of active segments for this flag
    valid : `~segments.SegmentList`, optional
        A list of valid segments for this flag
    """
    _EntryClass = Segment
    _ListClass = SegmentList
    __slots__ = ('_active', '_valid', 'ifo', 'name', 'version',
                 'category', 'comment', '_name_is_flag')
    def __init__(self, name=None, active=[], valid=[], category=None,
                 comment=None):
        """Define a new DataQualityFlag, with a name, a set of active
        segments, and a set of valid segments
        """
        self.ifo, self.name, self.version = parse_flag_name(name)
        if self.ifo:
            self._name_is_flag = True
        else:
            self._name_is_flag = False
        self.valid = valid
        self.active = active
        self.category = category
        self.comment = comment

    @property
    def active(self):
        """The set of segments during which this DataQualityFlag was
        active.
        """
        if self.valid:
            self._active &= self.valid
            self._active.coalesce()
        return self._active

    @active.setter
    def active(self, segmentlist):
        self._active = self._ListClass(map(self._EntryClass,
                                          segmentlist)).coalesce()
        if not self.valid and len(segmentlist):
            self.valid = [segmentlist.extent()]

    @property
    def valid(self):
        """The set of segments during which this DataQualityFlag was
        valid, and its state was well defined.
        """
        return self._valid

    @valid.setter
    def valid(self, segmentlist):
        self._valid = self._ListClass(map(self._EntryClass,
                                         segmentlist)).coalesce()

    @property
    def extent(self):
        """The single GPS [start, stop) enclosing segment of this
        `DataQualityFlag`
        """
        return self.valid.extent()

    def __getitem__(self, item):
        if isinstance(item, int) or isinstance(item, slice):
            return self.active[item]
        else:
            new = self.__class__()
            new.ifo = self.ifo
            new.name = self.name
            new.version = self.version
            new.active = self.active[item]
            new.valid = self.valid
            return new

    @classmethod
    def query(cls, flag, *args, **kwargs):
        """Query the segment database URL as give, returning segments
        during which the given flag was defined and active.

        Parameters
        ----------
        flag : str
            The name of the flag for which to query
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a
            :class:`~gwpy.segments.segments.SegmentList`
            defining a number of summary segments
        url : `str`, optional, default: ``'https://segdb.ligo.caltech.edu'``
            URL of the segment database, defaults to segdb.ligo.caltech.edu

        Returns
        -------
        flag : `DataQualityFlag`
            A new `DataQualityFlag`, with the `valid` and `active` lists
            filled appropriately.
        """
        # parse arguments
        if len(args) == 1 and isinstance(args[0], SegmentList):
            qsegs = args[0]
        elif len(args) == 1 and len(args[0]) == 2:
            qsegs = SegmentList(Segment(args[0]))
        elif len(args) == 2:
            qsegs = SegmentList([Segment(args)])
        else:
            raise ValueError("DataQualityFlag.query must be called with a "
                             "flag name, and either GPS start and stop times, "
                             "or a SegmentList of query segments")
        # process query
        try:
            flags = DataQualityDict.query([flag], qsegs, **kwargs)
        except TypeError as e:
            if 'DataQualityDict' in str(e):
                raise TypeError(str(e).replace('DataQualityDict',
                                               cls.__name__))
            else:
                raise
        if len(flags) != 1:
            raise RuntimeError("Multiple flags returned for single query, "
                               "something went wrong")
        return flags[flag]

    read = classmethod(io_registry.read)

    def round(self):
        """Expand this `DataQualityFlag` to integer segments, with each
        segment protracted to the nearest integer boundaries
        """
        new = self.copy()
        new.active = self._ListClass([self._EntryClass(int(floor(s[0])),
                                                       int(ceil(s[1]))) for
                                     s in new.active])
        new.valid = self._ListClass([self._EntryClass(int(floor(s[0])),
                                                      int(ceil(s[1]))) for
                                     s in new.valid])
        return new.coalesce()

    def coalesce(self):
        """Run the coalesce method over both the active and valid
        SegmentLists
        """
        self.active = self.active.coalesce()
        self.valid = self.valid.coalesce()
        return self

    def __repr__(self):
        indent = " " * len("<%s(" % self.__class__.__name__)
        valid = str(self.valid).replace("\n",
                                         "\n%s      " % indent)
        active = str(self.active).replace("\n",
                                           "\n%s       " % indent)
        return ("<{1}(valid={2},\n{0}active={3},\n{0}ifo={4},\n{0}name={5},\n"
                "{0}version={6},\n{0}comment={7})>".format(
                    indent, self.__class__.__name__, valid, active,
                    repr(self.ifo), repr(self.name), repr(self.version),
                    repr(self.comment)))

    def copy(self):
        """Build an exact copy of this `DataQualityFlag`, with a
        fresh memory copy of all segments and metadata.
        """
        new = self.__class__()
        new.ifo = self.ifo
        new.name = self.name
        new.version = self.version
        new.comment = self.comment
        new.valid = self._ListClass([self._EntryClass(s[0], s[1]) for
                                    s in self.valid])
        new.active = self._ListClass([self._EntryClass(s[0], s[1]) for
                                     s in self.active])
        return new

    def __and__(self, other):
        """Return a new `DataQualityFlag` that is the intersection of
        this one and ``other``
        """
        return self.copy().__iand__(other)

    def __iand__(self, other):
        """Replace this `DataQualityFlag` with the intersection of
        itself and ``other``
        """
        self.valid &= other.valid
        self.active &= other.active
        return self

    def __sub__(self, other):
        """Return a new `DataQualityFlag` that is the union of this
        one and ``other``
        """
        return self.copy().__isub__(other)

    def __isub__(self, other):
        """Replace this `DataQualityFlag` with the difference between
        itself and ``other``
        """
        self.valid -= other.valid
        self.active -= other.active
        return self

    def __or__(self, other):
        """Return a new `DataQualityFlag` that is the union of this
        one and ``other``
        """
        return self.copy().__ior__(other)

    def __ior__(self, other):
        """Replace this `DataQualityFlag` with the union of itself
        and ``other``
        """
        self.valid |= other.valid
        self.active |= other.active
        return self

    __add__ = __or__
    __iadd__ = __ior__

    def plot(self, **kwargs):
        """Plot this DataQualityFlag
        """
        from ..plotter import SegmentPlot
        kwargs.setdefault('epoch', self.valid[0][0])
        return SegmentPlot(self, **kwargs)

    write = io_registry.write

class DataQualityDict(OrderedDict):
    """List of `DataQualityFlags` with associated methods
    """
    _EntryClass = DataQualityFlag

    @classmethod
    def query(cls, flags, *args, **kwargs):
        """Query the segment database URL as given for the listed
        `DataQualityFlag` names

        Parameters
        ----------
        flag : str
            The name of the flag for which to query
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a
            :class:`~gwpy.segments.segments.SegmentList`
            defining a number of summary segments
        url : `str`, optional, default: ``'https://segdb.ligo.caltech.edu'``
            URL of the segment database, defaults to segdb.ligo.caltech.edu

        Returns
        -------
        flag : `DataQualityFlag`
            A new `DataQualityFlag`, with the `valid` and `active` lists
            filled appropriately.
        """
        # given segmentlist
        if len(args) == 1 and isinstance(args[0], SegmentList):
            qsegs = args[0]
        elif len(args) == 1 and len(args[0]) == 2:
            qsegs = SegmentList(Segment(args[0]))
        elif len(args) == 2:
            qsegs = SegmentList([Segment(args)])
        else:
            raise ValueError("DataQualityDict.query must be called with a "
                             "flag name, and either GPS start and stop times, "
                             "or a SegmentList of query segments")
        url = kwargs.pop('url', 'https://segdb.ligo.caltech.edu')
        if kwargs.keys():
            raise TypeError("DataQualityDict.query has no keyword argument "
                            "'%s'" % kwargs.keys()[0])
        # parse flags
        if isinstance(flags, basestring):
            flags = flags.split(',')
        else:
            flags = flags
        # process query
        from glue.segmentdb import (segmentdb_utils as segdb_utils,
                                    query_engine as segdb_engine)
        connection = segdb_utils.setup_database(url)
        engine = segdb_engine.LdbdQueryEngine(connection)
        segdefs = []
        for flag in flags:
            ifo, name, version = parse_flag_name(flag)
            if not version:
                version = '*'
            for gpsstart, gpsend in qsegs:
                gpsstart = int(float(gpsstart))
                gpsend = int(ceil(float(gpsend)))
                segdefs += segdb_utils.expand_version_number(
                               engine, (ifo, name, version,
                                        gpsstart, gpsend, 0, 0))
        segs = segdb_utils.query_segments(engine, 'segment', segdefs)
        segsum = segdb_utils.query_segments(engine, 'segment_summary', segdefs)
        segs = [s.coalesce() for s in segs]
        segsum = [s.coalesce() for s in segsum]
        # build output
        out = cls()
        for definition, segments, summary in zip(segdefs, segs, segsum):
            # parse flag name
            flag = ':'.join(map(str, definition[:3]))
            if flag.endswith('*'):
                flag = flag.rsplit(':', 1)[0]
            # define flag
            if not flag in out:
                out[flag] = DataQualityFlag(name=flag)
            # add segments
            out[flag].valid.extend(summary)
            out[flag].active.extend(segments)
        return out

    read = classmethod(io_registry.read)

    # -----------------------------------------------------------------------
    # DataQualityDict operators

    def __iand__(self, other):
        for key, value in other.iteritems():
            if key in self:
                self[key] &= value
            else:
                self[key] = self._EntryClass()
        return self

    def __and__(self, other):
        if sum(len(s) for s in self.values()) <= sum(len(s) for s in
                                                     other.values()):
            return self.copy().__iand__(other)
        return other.copy().__iand__(self)

    def __ior__(self, other):
        for key, value in other.iteritems():
            if key in self:
                self[key] |= value
            else:
                self[key] = shallowcopy(value)
        return self

    def __or__(self, other):
        if sum(len(s) for s in self.values()) >= sum(len(s) for s in
                                                     other.values()):
            return self.copy().__ior__(other)
        return other.copy().__ior__(self)

    __iadd__ = __ior__
    __add__ = __or__

    def __isub__(self, other):
        for key, value in other.iteritems():
            if key in self:
                self[key] -= value
        return self

    def __sub__(self, other):
        return self.copy().__isub__(other)

    def __ixor__(self, other):
        for key, value in other.iteritems():
            if key in self:
                self[key] ^= value
            else:
                self[key] = shallowcopy(value)
        return self

    def __xor__(self, other):
        if sum(len(s) for s in self.values()) <= sum(len(s) for s in
                                                     other.values()):
            return self.copy().__ixor__(other)
        return other.copy().__ixor__(self)

    def __invert__(self):
        new = self.copy()
        for key, value in new.items():
            dict.__setitem__(new, key, ~value)
        return new


_re_inv = re.compile(r"\A(?P<ifo>[A-Z]\d):(?P<name>[^/]+):(?P<version>\d+)\Z")
_re_in = re.compile(r"\A(?P<ifo>[A-Z]\d):(?P<name>[^/]+)\Z")
_re_nv = re.compile(r"\A(?P<name>[^/]+):(?P<ver>\d+)\Z")

def parse_flag_name(name, warn=True):
    """Internal method to parse a `string` name into constituent
    `ifo, `name` and `version` components.

    Parameters
    ----------
    name : `str`
        the full name of a `DataQualityFlag` to parse, e.g.
        ``'H1:DMT-SCIENCE:1'``

    Returns
    -------
    (ifo, name, version)
        A tuple of component string parts
    """
    if name is None:
        return None, None, None
    if _re_inv.match(name):
        match = _re_inv.match(name).groupdict()
        return match['ifo'], match['name'], int(match['version'])
    elif _re_in.match(name):
        match = _re_in.match(name).groupdict()
        return match['ifo'], match['name'], None
    elif _re_nv.match(name):
        match = _re_nv.match(name).groupdict()
        return None, match['name'], int(match['version'])
    if warn:
        warnings.warn("No flag name structure detected in '%s', flags should "
                      "be named as 'IFO:DQ-FLAG_NAME:VERSION'" % name)
    return None, name, None
