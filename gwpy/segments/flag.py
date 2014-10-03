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

"""This module defines the `DataQualityFlag` and `DataQualityDict`.

The `DataQualityFlag` represents an annotated set of data-quality segments
indicating something about the state of a laser-interferometer
gravitational-wave detector in a given time interval.

The `DataQualityDict` is just a `dict` of flags, provided as a convenience
for handling multiple flags over the same global time interval.
"""

import operator
import re
from urlparse import urlparse
from copy import copy as shallowcopy
from math import (floor, ceil)

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from glue.ligolw import utils as ligolw_utils
from glue.ligolw.lsctables import VetoDefTable

from .. import version
from ..utils.deps import with_import
from ..io import (reader, writer)
from .segments import Segment, SegmentList

__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['DataQualityFlag', 'DataQualityDict']

_re_inv = re.compile(r"\A(?P<ifo>[A-Z]\d):(?P<tag>[^/]+):(?P<version>\d+)\Z")
_re_in = re.compile(r"\A(?P<ifo>[A-Z]\d):(?P<tag>[^/]+)\Z")
_re_nv = re.compile(r"\A(?P<tag>[^/]+):(?P<ver>\d+)\Z")


class DataQualityFlag(object):
    """A representation of a named set of segments.

    Parameters
    ----------
    name : str, optional
        The name of this `DataQualityFlag`.
        This should be of the form {ifo}:{tag}:{version}, e.g.
        'H1:DMT-SCIENCE:1'. Use `label` for human-readable names.
    active : :class:`~gwpy.segments.segments.SegmentList`, optional
        A list of active segments for this flag
    valid : :class:`~gwpy.segments.segments.SegmentList`, optional
        A list of valid segments for this flag
    label : `str`, optional
        Human-readable name for this flag, e.g. ``'Science-mode'``
    category : `int`, optional
        Veto category for this flag.
    description : `str`, optional
        Human-readable description of why this flag was created.
    isgood : `bool`, optional
        Do active segments mean the IFO was in a good state?
    """
    _EntryClass = Segment
    _ListClass = SegmentList

    def __init__(self, name=None, active=None, valid=None, label=None,
                 category=None, description=None, isgood=True, padding=None):
        """Define a new DataQualityFlag.
        """
        self.name = name
        self.valid = valid or []
        self.active = active or []
        self.label = label
        self.category = category
        self.description = description
        self.isgood = isgood
        self.padding = padding

    # -------------------------------------------------------------------------
    # read-write properties

    @property
    def name(self):
        """The name associated with this `DataQualityFlag`.

        This normally takes the form {ifo}:{tag}:{version}. If found,
        each component is stored separately the associated attributes.

        :type: `str`
        """
        return self._name

    @name.setter
    def name(self, n):
        self._name = n
        try:
            self._parse_name(n)
        except ValueError:
            self._parse_name(None)

    @property
    def ifo(self):
        """The interferometer associated with this `DataQualityFlag`.

        This should be a single uppercase letter and a single number,
        e.g. ``'H1'``.

        :type: `str`
        """
        return self._ifo

    @ifo.setter
    def ifo(self, ifoname):
        self._ifo = ifoname

    @property
    def tag(self):
        """The tag (name) associated with this `DataQualityFlag`.

        This should take the form ``'AAA-BBB_CCC_DDD'``, i.e. where
        each component is an uppercase acronym of alphanumeric
        characters only, e.g. ``'DCH-IMC_BAD_CALIBRATION'`` or
        ``'DMT-SCIENCE'``.

        :type: `str`
        """
        return self._tag

    @tag.setter
    def tag(self, n):
        self._tag = n

    @property
    def version(self):
        """The version number of this `DataQualityFlag`.

        Each flag in the segment database is stored with a version
        integer, with each successive version representing a more
        accurate dataset for its valid segments than any previous.

        :type: `str`
        """
        return self._version

    @version.setter
    def version(self, v):
        if v is None:
            self._version = None
        else:
            self._version = int(v)

    @property
    def label(self):
        """A human-readable label for this `DataQualityFlag`.

        For example: ``'Science-mode'``.

        :type: `str`
        """
        return self._label

    @label.setter
    def label(self, lab):
        self._label = lab

    @property
    def active(self):
        """The set of segments during which this `DataQualityFlag` was
        active.
        """
        if self.valid:
            self._active &= self.valid
        return self._active

    @active.setter
    def active(self, segmentlist):
        self._active = self._ListClass(map(self._EntryClass,
                                           segmentlist))
        if not self.valid and len(segmentlist):
            self.valid = [segmentlist.extent()]

    @property
    def valid(self):
        """The set of segments during which this `DataQualityFlag` was
        valid, and its state was well defined.
        """
        return self._valid

    @valid.setter
    def valid(self, segmentlist):
        self._valid = self._ListClass(map(self._EntryClass,
                                          segmentlist))

    @property
    def category(self):
        """Veto category for this `DataQualityFlag`.

        :type: `int`
        """
        return self._category

    @category.setter
    def category(self, cat):
        if cat is None:
            self._category = None
        else:
            self._category = int(cat)

    @property
    def description(self):
        """Description of why/how this `DataQualityFlag` was generated.

        :type: `str`
        """
        return self._description

    @description.setter
    def description(self, desc):
        self._description = desc

    @property
    def isgood(self):
        """Whether `active` segments mean the instrument was in a good state.

        :type: `bool`
        """
        return self._isgood

    @isgood.setter
    def isgood(self, good):
        self._isgood = bool(good)

    @property
    def padding(self):
        """[start, end) padding for this flag's active segments.
        """
        try:
            return self._padding
        except AttributeError:
            self._padding = Segment(0, 0)
            return self.padding


    @padding.setter
    def padding(self, pad):
        if pad is None:
            self._padding = Segment(0, 0)
        else:
            self._padding = Segment(pad[0], pad[1])

    # -------------------------------------------------------------------------
    # read-only properties

    @property
    def texname(self):
        """Name of this `DataQualityFlag` in LaTeX printable format.
        """
        try:
            return self.name.replace('_', r'\_')
        except AttributeError:
            return None

    @property
    def extent(self):
        """The single GPS ``[start, stop)`` enclosing segment of this
        `DataQualityFlag`.

        :type: :class:`~gwpy.segments.segment.Segment`
        """
        return self.valid.extent()

    @property
    def livetime(self):
        """Amount of time this `DataQualityFlag` was `active`.

        :type: `float`
        """
        return abs(self.active)

    # -------------------------------------------------------------------------
    # classmethods

    @classmethod
    def query(cls, flag, *args, **kwargs):
        """Query the segment database for the given flag.

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
            URL of the segment database

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

    @classmethod
    @with_import('dqsegdb.apicalls')
    def query_dqsegdb(cls, flag, *args, **kwargs):
        """Query the DQSegDB for the given flag.

        Parameters
        ----------
        flag : str
            The name of the flag for which to query
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a
            :class:`~gwpy.segments.segments.SegmentList`
            defining a number of summary segments
        url : `str`, optional, default: ``'https://dqsegdb.ligo.org'``
            URL of the segment database

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
        # get server
        protocol, server = kwargs.pop(
            'url', 'https://dqsegdb.ligo.org').split('://', 1)

        # parse flag
        try:
            ifo, name, version = flag.split(':', 2)
        except ValueError as e:
            e.args = ('Flag must be of the form \'IFO:FLAG-NAME:VERSION\'',)
            raise
        else:
            version = int(version)

        # other keyword arguments
        request = kwargs.pop('request', 'metadata,active,known')

        # process query
        new = cls(name=flag)
        for seg in qsegs:
            data, uri = apicalls.dqsegdbQueryTimes(protocol, server, ifo,
                                                   name, version, request,
                                                   seg[0], seg[1])
            for seg in data['active']:
                new.active.append(Segment(*seg))
            for seg in data['known']:
                new.valid.append(Segment(*seg))
            new.description = data['metadata']['comment']
            new.isgood = not data['metadata']['active_indicates_ifo_badness']

        return new

    # use input/output registry to allow multi-format reading
    read = classmethod(reader(doc="""
    Read segments from file into a `DataQualityFlag`.

    Parameters
    ----------
    filename : `str`
        path of file to read
    format : `str`, optional
        source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.
    flag : `str`, optional, default: read all segments
        name of flag to read from file.
    coltype : `type`, optional, default: `float`
        datatype to force for segment times, only valid for
        ``format='segwizard'``.
    strict : `bool`, optional, default: `True`
        require segment start and stop times match printed duration,
        only valid for ``format='segwizard'``.


    Returns
    -------
    dqflag : `DataQualityFlag`
        formatted `DataQualityFlag` containing the active and valid
        segments read from file.

    Notes
    -----
    When reading with ``format='segwizard'`` the
    :attr:`~DataQualityFlag.valid` `SegmentList` will simply represent
    the extent of the :attr:`~DataQualityFlag.active` `SegmentList`.
    """))

    @classmethod
    def from_veto_def(cls, veto):
        """Define a new `DataQualityFlag` from a :class:`~glue.ligolw.lsctables.VetoDef.`
        """
        name = '%s:%s' % (veto.ifo, veto.name)
        try:
            name += ':%d' % int(veto.version)
        except TypeError:
            pass
        valid = Segment(veto.start_time, veto.end_time)
        pad = Segment(veto.start_pad, veto.end_pad)
        return cls(name=name, valid=[valid], category=veto.category,
                   description=veto.comment, padding=pad)

    # -------------------------------------------------------------------------
    # instance methods

    write = writer()

    def populate(self, source='https://segdb.ligo.caltech.edu', segments=None,
                 **kwargs):
        """Query the segment database for this flag's active segments.

        This method assumes all of the metadata for each flag have been
        filled. Minimally, the following attributes must be filled

        .. autosummary::

           ~DataQualityFlag.name
           ~DataQualityFlag.valid

        Segments will be fetched from the database, with any
        :attr:`~DataQualityFlag.padding` added on-the-fly.

        This `DataQualityFlag` will be modified in-place.

        Parameters
        ----------
        source : `str`
            source of segments for this `DataQualityFlag`. This must be
            either a URL for a segment database or a path to a file on disk.
        segments : `SegmentList`, optional
            a list of valid segments during which to query, if not given,
            existing valid segments for this flag will be used.
        **kwargs
            any other keyword arguments to be passed to
            :meth:`DataQualityFlag.query` or :meth:`DataQualityFlag.read`.

        Returns
        -------
        self : `DataQualityFlag`
            a reference to this flag
        """
        tmp = DataQualityDict()
        tmp[self.name] = self
        tmp.populate(source=source, segments=segments, **kwargs)
        return tmp[self.name]

    def contract(self, x):
        """Contract each of the `active` `Segments` by ``x`` seconds.

        This method adds ``x`` to each segment's lower bound, and subtracts
        ``x`` from the upper bound.

        The :attr:`~DataQualityFlag.active` `SegmentList` is modified
        in place.

        Parameters
        ----------
        x : `float`
            number of seconds by which to contract each `Segment`.
        """
        self.active = self.active.contract(x)
        return self.active

    def protract(self, x):
        """Protract each of the `active` `Segments` by ``x`` seconds.

        This method subtracts ``x`` from each segment's lower bound,
        and adds ``x`` to the upper bound, while maintaining that each
        `Segment` stays within the `valid` bounds.

        The :attr:`~DataQualityFlag.active` `SegmentList` is modified
        in place.

        Parameters
        ----------
        x : `float`
            number of seconds by which to protact each `Segment`.
        """
        self.active = self.active.protract(x)
        return self.active

    def round(self):
        """Round this `DataQualityFlag` to integer segments.

        Returns
        -------
        roundedflag : `DataQualityFlag`
            A copy of the original flag with the `active` and `valid` segments
            padded out to the enclosing integer boundaries.
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
        """Coalesce the segments for this `DataQualityFlag`.

        This method calls the
        :meth:`~gwpy.segments.segments.SegmentList.coalesce` method of the
        underlying `active` and `valid` segment lists.

        .. note::

            this operations is performed in-place.

        Returns
        -------
        self
            a view of this flag, not a copy.
        """
        self.active = self.active.coalesce()
        self.valid = self.valid.coalesce()
        return self

    def __repr__(self):
        indent = " " * len("<%s(" % self.__class__.__name__)
        valid = str(self.valid).replace("\n",
                                        "\n%s      " % indent).split('\n')
        if len(valid) > 10:
            valid = valid[:3] + ['%s      ...' % indent] + valid[-3:]
        active = str(self.active).replace("\n",
                                          "\n%s       " % indent).split('\n')
        if len(active) > 10:
            active = active[:3] + ['%s        ...' % indent] + active[-3:]
        return ("<{1}({2},\n{0}valid={3},\n{0}active={4},\n"
                "{0}description={5})>".format(indent, self.__class__.__name__,
                                              (self.name and repr(self.name) or
                                               'No name'),
                                              '\n'.join(valid),
                                              '\n'.join(active),
                                              repr(self.description)))

    def copy(self):
        """Build an exact copy of this `DataQualityFlag`.

        Returns
        -------
        flag2 : `DataQualityFlag`
            a copy of the original flag, but with a fresh memory address.
        """
        new = self.__class__()
        new.ifo = self.ifo
        new.name = self.name
        new.version = self.version
        new.description = self.description
        new.valid = self._ListClass([self._EntryClass(s[0], s[1]) for
                                    s in self.valid])
        new.active = self._ListClass([self._EntryClass(s[0], s[1]) for
                                     s in self.active])
        return new

    def plot(self, **kwargs):
        """Plot this `DataQualityFlag`.

        Parameters
        ----------
        **kwargs
            all keyword arguments are passed to the
            :class:`~gwpy.plotter.segments.SegmentPlot` constructor.

        Returns
        -------
        plot : `~gwpy.plotter.segments.SegmentPlot`
            a new `Plot` with this `DataQualityFlag` displayed on a set of
            :class:`~gwpy.plotter.segments.SegmentAxes`.
        """
        from ..plotter import (rcParams, SegmentPlot)
        kwargs.setdefault('epoch', self.valid[0][0])
        if self.label:
            kwargs.setdefault('label', self.label)
        elif rcParams['text.usetex']:
            kwargs.setdefault('label', self.texname)
        else:
            kwargs.setdefault('label', self.name)
        return SegmentPlot(self, **kwargs)

    def _parse_name(self, name):
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

        Raises
        ------
        `ValueError`
            If the input ``name`` cannot be parsed into
            {ifo}:{tag}:{version} format.
        """
        if name is None:
            self.ifo = None
            self.tag = None
            self.version = None
        elif _re_inv.match(name):
            match = _re_inv.match(name).groupdict()
            self.ifo = match['ifo']
            self.tag = match['tag']
            self.version = int(match['version'])
        elif _re_in.match(name):
            match = _re_in.match(name).groupdict()
            self.ifo = match['ifo']
            self.tag = match['tag']
            self.version = None
        elif _re_nv.match(name):
            match = _re_nv.match(name).groupdict()
            self.ifo = None
            self.tag = match['tag']
            self.version = None
        else:
            raise ValueError("No flag name structure detected in '%s', flags "
                             "should be named as '{ifo}:{tag}:{version}'. "
                             "For arbitrary strings, use the "
                             "`DataQualityFlag.label` attribute" % name)
        return self.ifo, self.tag, self.version

    def __getslice__(self, slice_):
        return self.__class__(name=self.name, valid=self.valid,
                              active=self.active[slice_], label=self.label,
                              description=self.description, isgood=self.isgood)

    def __getitem__(self, item):
        if isinstance(item, int) or isinstance(item, slice):
            return self.active[item]
        else:
            self.__getslice__(item)

    def __and__(self, other):
        """Find the intersection of this one and ``other``.
        """
        return self.copy().__iand__(other)

    def __iand__(self, other):
        """Intersect this `DataQualityFlag` with ``other`` in-place.
        """
        self.valid &= other.valid
        self.active &= other.active
        return self

    def __sub__(self, other):
        """Find the difference between this `DataQualityFlag` and another.
        """
        return self.copy().__isub__(other)

    def __isub__(self, other):
        """Subtract the ``other`` `DataQualityFlag` from this one in-place.
        """
        #self.valid -= other.valid
        self.active -= other.active
        return self

    def __or__(self, other):
        """Find the union of this `DataQualityFlag` and ``other``.
        """
        return self.copy().__ior__(other)

    def __ior__(self, other):
        """Add the ``other`` `DataQualityFlag` to this one in-place.
        """
        self.valid |= other.valid
        self.active |= other.active
        return self

    __add__ = __or__
    __iadd__ = __ior__


class DataQualityDict(OrderedDict):
    """An `OrderedDict` of (key, `DataQualityFlag`) pairs.

    Since the `DataQualityDict` is an `OrderedDict`, all iterations over
    its elements retain the order in which they were inserted.
    """
    _EntryClass = DataQualityFlag

    # -----------------------------------------------------------------------
    # classmethods

    @classmethod
    def query(cls, flags, *args, **kwargs):
        """Query the segment database for a set of `DataQualityFlags`.

        Parameters
        ----------
        flags : `iterable`
            A list of flag names for which to query.
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a
            :class:`~gwpy.segments.segments.SegmentList`
            defining a number of summary segments.
        url : `str`, optional, default: ``'https://segdb.ligo.caltech.edu'``
            URL of the segment database.

        Returns
        -------
        flagdict : `DataQualityDict
            An ordered `DataQualityDict` of (name, `DataQualityFlag`)
            pairs.
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
        if isinstance(flags, (str, unicode)):
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
            dqflag = DataQualityFlag(name=flag)
            ifo = dqflag.ifo
            name = dqflag.tag
            if dqflag.version is None:
                vers = '*'
            else:
                vers = dqflag.version
            for gpsstart, gpsend in qsegs:
                gpsstart = float(gpsstart)
                if not gpsstart.is_integer():
                    raise ValueError("Segment database queries can only"
                                     "operate on integer GPS times")
                gpsend = float(gpsend)
                if not gpsend.is_integer():
                    raise ValueError("Segment database queries can only"
                                     "operate on integer GPS times")
                segdefs += segdb_utils.expand_version_number(
                    engine, (ifo, name, vers, gpsstart, gpsend, 0, 0))
        segs = segdb_utils.query_segments(engine, 'segment', segdefs)
        segsum = segdb_utils.query_segments(engine, 'segment_summary', segdefs)
        segs = [s.coalesce() for s in segs]
        segsum = [s.coalesce() for s in segsum]
        # build output
        out = cls()
        for definition, segments, summary in zip(segdefs, segs, segsum):
            # parse flag name
            flag = ':'.join(map(str, definition[:3]))
            name = flag.rsplit(':', 1)[0]
            # if versionless
            if flag.endswith('*'):
                flag = name
                key = name
            # if asked for versionless, but returned a version
            elif flag not in flags and name in flags:
                key = name
            # other
            else:
                key = flag
            # define flag
            if not key in out:
                out[key] = DataQualityFlag(name=flag)
            # add segments
            out[key].valid.extend(summary)
            out[key].active.extend(segments)
        return out

    def query_dqsegdb(cls, flags, *args, **kwargs):
        """Query the DQSegDB for a set of `DataQualityFlags`.

        Parameters
        ----------
        flags : `iterable`
            A list of flag names for which to query.
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a
            :class:`~gwpy.segments.segments.SegmentList`
            defining a number of summary segments.
        url : `str`, optional, default: ``'https://dqsegdb.ligo.org'``
            URL of the segment database.

        Returns
        -------
        flagdict : `DataQualityDict
            An ordered `DataQualityDict` of (name, `DataQualityFlag`)
            pairs.
        """
        new = cls()
        for flag in flags:
            new[flag] = _EntryClass.query_dqsegdb(flag, *args, **kwargs)
        return new

    # use input/output registry to allow multi-format reading
    read = classmethod(reader(doc="""
    Read segments from file into a `DataQualityDict`.

    Parameters
    ----------
    filename : `str`
        path of file to read
    format : `str`, optional
        source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.
    flags : `list`, optional, default: read all flags found
        list of flags to read, by default all flags are read separately.
    coalesce : `bool`, optional, default: `True`
        coalesce all `SegmentLists` before returning.

    Returns
    -------
    flagdict : :class:`~gwpy.segments.flag.DataQualityDict`
        a new `DataQualityDict` of `DataQualityFlag` entries with ``active``
        and ``valid`` segments seeded from the XML tables in the given
        file.

    Notes
    -----
    """))

    @classmethod
    def from_veto_definer_file(cls, fp, start=None, end=None, ifo=None):
        """Read a `DataQualityDict` from a LIGO_LW XML VetoDefinerTable.
        """
        # open file
        if isinstance(fp, (str, unicode)):
            fobj = open(fp, 'r')
        else:
            fobj = fp
        xmldoc = ligolw_utils.load_fileobj(fobj)[0]
        # read veto definers
        veto_def_table = VetoDefTable.get_table(xmldoc)
        out = cls()
        for row in veto_def_table:
            if ifo and row.ifo != ifo:
                continue
            if start and 0 < row.end_time < start:
                continue
            elif start:
                row.start_time = max(row.start_time, start)
            if end and row.start_time > end:
                continue
            elif end and not row.end_time:
                row.end_time = end
            elif end:
                row.end_time = min(row.end_time, end)
            flag = DataQualityFlag.from_veto_def(row)
            if flag.name in out:
                out[flag.name].valid.extend(flag.valid)
                out[flag.name].valid.coalesce()
            else:
                out[flag.name] = flag
        return out

    # -----------------------------------------------------------------------
    # instance methods

    def populate(self, source='https://segdb.ligo.caltech.edu',
                 segments=None, **kwargs):
        """Query the segment database for each flag's active segments.

        This method assumes all of the metadata for each flag have been
        filled. Minimally, the following attributes must be filled

        .. autosummary::

           ~DataQualityFlag.name
           ~DataQualityFlag.valid

        Segments will be fetched from the database, with any
        :attr:`~DataQualityFlag.padding` added on-the-fly.

        Entries in this `DataQualityDict` will be modified in-place.

        Parameters
        ----------
        source : `str`
            source of segments for this `DataQualityFlag`. This must be
            either a URL for a segment database or a path to a file on disk.
        segments : `SegmentList`, optional
            a list of valid segments during which to query, if not given,
            existing valid segments for flags will be used.
        **kwargs
            any other keyword arguments to be passed to
            :meth:`DataQualityFlag.query` or :meth:`DataQualityFlag.read`.

        Returns
        -------
        self : `DataQualityDict`
            a reference to the modified DataQualityDict
        """
        # format source
        source = urlparse(source)
        valid = reduce(operator.or_,
                       [f.valid for f in self.values()]).coalesce()
        if segments:
            valid &= SegmentList(segments)
        if source.netloc:
            tmp = type(self).query(self.keys(), valid,
                                   url=source.geturl(), **kwargs)
        else:
            tmp = type(self).read(source.geturl(), self.name, **kwargs)
        for key, flag in self.iteritems():
            self[key].valid &= valid
            self[key].active = [type(seg)(seg[0] - flag.padding[0],
                                          seg[1] + flag.padding[1])
                                for seg in tmp[key].active]
        return self

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
        if (sum(len(s.active) for s in self.values()) >=
            sum(len(s.active) for s in other.values())):
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

    def union(self):
        """Return the union of all flags in this `DataQualityDict`

        Returns
        -------
        union : `DataQualityFlag`
            a new `DataQualityFlag` who's active and valid segments
            are the union of those of the values of this `DataQualityDict`.
        """
        usegs = reduce(operator.or_, self.itervalues())
        usegs.name = ' | '.join(self.iterkeys())
        return usegs

    def intersection(self):
        """Return the intersection of all flags in this `DataQualityDict`

        Returns
        -------
        intersection : `DataQualityFlag`
            a new `DataQualityFlag` who's active and valid segments
            are the intersection of those of the values of this
            `DataQualityDict`.
        """
        isegs = reduce(operator.and_, self.itervalues())
        isegs.name = ' & '.join(self.iterkeys())
        return isegs

    def plot(self, label='key', **kwargs):
        """Plot the data for this `DataQualityDict`.

        Parameters
        ----------
        label : `str`, optional
            labelling system to use, or fixed label for all `DataQualityFlags`.
            Special values include

            - ``'key'``: use the key of the `DataQualityDict`,
            - ``'name'``: use the :attr:`~DataQualityFlag.name` of the
              `DataQualityFlag`

            If anything else, that fixed label will be used for all lines.

        **kwargs
            all other keyword arguments are passed to the plotter as
            appropriate

        See Also
        --------
        gwpy.plotter.SegmentPlot
        gwpy.plotter.SegmentAxes
        gwpy.plotter.SegmentAxes.plot_dqdict
        """
        from ..plotter import SegmentPlot
        figargs = dict()
        for key in ['figsize', 'dpi']:
            if key in kwargs:
                figargs[key] = kwargs.pop(key)
        axargs = dict()
        for key in ['insetlabels']:
            if key in kwargs:
                axargs[key] = kwargs.pop(key)
        plot_ = SegmentPlot(**figargs)
        ax = plot_.gca(**axargs)
        ax.plot(self, label=label, **kwargs)
        ax.autoscale(axis='y')
        return plot_
