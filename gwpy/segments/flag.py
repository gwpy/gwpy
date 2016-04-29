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
import warnings
import tempfile
from urlparse import urlparse
from copy import (copy as shallowcopy, deepcopy)
from math import (floor, ceil)
from threading import Thread
from Queue import Queue

from six.moves.urllib import request
from six.moves.urllib.error import URLError

from numpy import inf

from glue.segments import PosInfinity

from ..time import to_gps
from ..utils.deps import with_import
from ..utils.compat import OrderedDict
from ..io import (reader, writer)
from .segments import Segment, SegmentList

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['DataQualityFlag', 'DataQualityDict']

re_IFO_TAG_VERSION = re.compile(r"\A(?P<ifo>[A-Z]\d):(?P<tag>[^/]+):(?P<version>\d+)\Z")
re_IFO_TAG = re.compile(r"\A(?P<ifo>[A-Z]\d):(?P<tag>[^/]+)\Z")
re_TAG_VERSION = re.compile(r"\A(?P<tag>[^/]+):(?P<version>\d+)\Z")


class DataQualityFlag(object):
    """A representation of a named set of segments.

    Parameters
    ----------
    name : str, optional
        The name of this flag.
        This should be of the form {ifo}:{tag}:{version}, e.g.
        'H1:DMT-SCIENCE:1'. Use `label` for human-readable names.

    active : `SegmentList`, optional
        A list of active segments for this flag

    known : `SegmentList`, optional
        A list of known segments for this flag

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

    def __init__(self, name=None, active=None, known=None, label=None,
                 category=None, description=None, isgood=True, padding=None,
                 valid=None):
        """Define a new DataQualityFlag.
        """
        self.name = name
        if valid is not None:
            if known is not None:
                raise ValueError("Please give only 'known', and not both "
                                 "'known' and 'valid'")
            self.valid = valid
        else:
            self.known = known
        self.active = active
        self.label = label
        self.category = category
        self.description = description
        self.isgood = isgood
        self.padding = padding

    # -------------------------------------------------------------------------
    # read-write properties

    @property
    def name(self):
        """The name associated with this flag.

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
        """The interferometer associated with this flag.

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
        """The tag (name) associated with this flag.

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
        """The version number of this flag.

        Each flag in the segment database is stored with a version
        integer, with each successive version representing a more
        accurate dataset for its known segments than any previous.

        :type: `int`
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
        """A human-readable label for this flag.

        For example: ``'Science-mode'``.

        :type: `str`
        """
        return self._label

    @label.setter
    def label(self, lab):
        self._label = lab

    @property
    def active(self):
        """The set of segments during which this flag was
        active.
        """
        return self._active

    @active.setter
    def active(self, segmentlist):
        if segmentlist is None:
            del self.active
        else:
            self._active = self._ListClass(map(self._EntryClass, segmentlist))

    @active.deleter
    def active(self):
        self._active = self._ListClass()

    @property
    def known(self):
        """The set of segments during which this flag was
        known, and its state was well defined.
        """
        return self._known

    @known.setter
    def known(self, segmentlist):
        if segmentlist is None:
            del self.known
        else:
            self._known = self._ListClass(map(self._EntryClass, segmentlist))

    @known.deleter
    def known(self):
        self._known = self._ListClass()

    @property
    def valid(self):
        """The set of segments during which this flag was
        known, and its state was well defined.
        """
        warnings.warn("The 'valid' property of the DataQualityFlag "
                      "has been renamed 'known' and will be removed in "
                      "the near future, please move to using 'known'.",
                      DeprecationWarning)
        return self.known

    @valid.setter
    def valid(self, segmentlist):
        warnings.warn("The 'valid' property of the DataQualityFlag "
                      "has been renamed 'known' and will be removed in "
                      "the near future, please move to using 'known'.",
                      DeprecationWarning)
        self.known = segmentlist

    @valid.deleter
    def valid(self):
        warnings.warn("The 'valid' property of the DataQualityFlag "
                      "has been renamed 'known' and will be removed in "
                      "the near future, please move to using 'known'.",
                      DeprecationWarning)
        self._known = self._ListClass()

    @property
    def category(self):
        """Veto category for this flag.

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
        """Description of why/how this flag was generated.

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
            self._padding = (0, 0)
            return self.padding

    @padding.setter
    def padding(self, pad):
        if pad is None:
            self._padding = (0, 0)
        else:
            self._padding = (pad[0], pad[1])

    # -------------------------------------------------------------------------
    # read-only properties

    @property
    def texname(self):
        """Name of this flag in LaTeX printable format.
        """
        try:
            return self.name.replace('_', r'\_')
        except AttributeError:
            return None

    @property
    def extent(self):
        """The single GPS ``[start, stop)`` enclosing segment of this
        `DataQualityFlag`.

        :type: `Segment`
        """
        return self.known.extent()

    @property
    def livetime(self):
        """Amount of time this flag was `active`.

        :type: `float`
        """
        return abs(self.active)

    @property
    def regular(self):
        """`True` if the `active` segments are a proper subset of the `known`.

        :type: `bool`
        """
        return abs(self.active - self.known) == 0

    # -------------------------------------------------------------------------
    # classmethods

    @classmethod
    def query(cls, flag, *args, **kwargs):
        """Query for segments of a given flag

        This method intelligently selects the `~DataQualityFlag.query_segdb`
        or the `~DataQualityFlag.query_dqsegdb` methods based on the
        ``url`` kwarg given.

        Parameters
        ----------
        flag : str
            The name of the flag for which to query
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a `SegmentList`
            defining a number of summary segments
        url : `str`, optional, default: ``'https://segments.ligo.org'``
            URL of the segment database

        See Also
        --------
        DataQualityFlag.query_segdb
        DataQualityFlag.query_dqsegdb
            for details on the actual query engine, and documentation of
            other keyword arguments appropriate for each query

        Returns
        -------
        flag : `DataQualityFlag`
            A new `DataQualityFlag`, with the `known` and `active` lists
            filled appropriately.
        """
        url = kwargs.get('url', 'https://segments.ligo.org')
        if 'dqsegdb' in url or re.match('https://[a-z1-9-]+.ligo.org', url):
            return cls.query_dqsegdb(flag, *args, **kwargs)
        else:
            return cls.query_segdb(flag, *args, **kwargs)

    @classmethod
    def query_segdb(cls, flag, *args, **kwargs):
        """Query the initial LIGO segment database for the given flag

        Parameters
        ----------
        flag : str
            The name of the flag for which to query
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a `SegmentList`
            defining a number of summary segments
        url : `str`, optional, default: ``'https://segments.ligo.org'``
            URL of the segment database

        Returns
        -------
        flag : `DataQualityFlag`
            A new `DataQualityFlag`, with the `known` and `active` lists
            filled appropriately.
        """
        # parse arguments
        if len(args) == 1 and isinstance(args[0], SegmentList):
            qsegs = args[0]
        elif len(args) == 1 and len(args[0]) == 2:
            qsegs = SegmentList(Segment(map(to_gps, args[0])))
        elif len(args) == 2:
            qsegs = SegmentList([Segment(map(to_gps, args))])
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
        if len(flags) > 1:
            raise RuntimeError("Multiple flags returned for single query, "
                               "something went wrong:\n    %s"
                               % '\n    '.join(flags.keys()))
        elif len(flags) == 0:
            raise RuntimeError("No flags returned for single query, "
                               "something went wrong.")
        return flags[flag]

    @classmethod
    @with_import('dqsegdb.apicalls')
    def query_dqsegdb(cls, flag, *args, **kwargs):
        """Query the advanced LIGO DQSegDB for the given flag

        Parameters
        ----------
        flag : str
            The name of the flag for which to query
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a `SegmentList`
            defining a number of summary segments
        url : `str`, optional, default: ``'https://segments.ligo.org'``
            URL of the segment database

        Returns
        -------
        flag : `DataQualityFlag`
            A new `DataQualityFlag`, with the `known` and `active` lists
            filled appropriately.
        """
        # parse arguments
        if len(args) == 1 and isinstance(args[0], SegmentList):
            qsegs = args[0]
        elif len(args) == 1 and len(args[0]) == 2:
            qsegs = SegmentList(Segment(map(to_gps, args[0])))
        elif len(args) == 2:
            qsegs = SegmentList([Segment(map(to_gps, args))])
        else:
            raise ValueError("DataQualityFlag.query must be called with a "
                             "flag name, and either GPS start and stop times, "
                             "or a SegmentList of query segments")
        # get server
        protocol, server = kwargs.pop(
            'url', 'https://segments.ligo.org').split('://', 1)

        # parse flag
        out = cls(name=flag)
        if out.ifo is None or out.tag is None:
            raise ValueError("Cannot parse ifo or tag (name) for flag %r"
                             % flag)

        # other keyword arguments
        request = kwargs.pop('request', 'metadata,active,known')

        # process query
        for start, end in qsegs:
            if end == PosInfinity or float(end) == +inf:
                end = to_gps('now').seconds
            if out.version is None:
                data, versions, _ = apicalls.dqsegdbCascadedQuery(
                    protocol, server, out.ifo, out.tag, request,
                    start, end)
                metadata = versions[-1]['metadata']
            else:
                try:
                    data, _ = apicalls.dqsegdbQueryTimes(
                        protocol, server, out.ifo, out.tag, out.version,
                        request, start, end)
                except URLError as e:
                    e.args = ('Error querying for %s: %s' % (flag, e),)
                    raise
                metadata = data['metadata']
            new = cls(name=flag)
            for s2 in data['active']:
                new.active.append(Segment(*s2))
            for s2 in data['known']:
                new.known.append(Segment(*s2))
            segl = SegmentList([Segment(start, end)])
            new.known &= segl
            new.active &= segl
            out += new
            out.description = metadata.get('flag_description', None)
            out.isgood = not metadata.get(
                'active_indicates_ifo_badness', False)

        return out

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
        formatted `DataQualityFlag` containing the active and known
        segments read from file.

    Notes
    -----
    When reading with ``format='segwizard'`` the
    :attr:`~DataQualityFlag.known` `SegmentList` will simply represent
    the extent of the :attr:`~DataQualityFlag.active` `SegmentList`.
    """))

    @classmethod
    def from_veto_def(cls, veto):
        """Define a `DataQualityFlag` from a `~glue.ligolw.lsctables.VetoDef`
        """
        name = '%s:%s' % (veto.ifo, veto.name)
        try:
            name += ':%d' % int(veto.version)
        except TypeError:
            pass
        if veto.end_time == 0:
            veto.end_time = PosInfinity
        known = Segment(veto.start_time, veto.end_time)
        pad = (veto.start_pad, veto.end_pad)
        return cls(name=name, known=[known], category=veto.category,
                   description=veto.comment, padding=pad)

    # -------------------------------------------------------------------------
    # instance methods

    write = writer()

    def populate(self, source='https://segments.ligo.org', segments=None,
                 pad=True, **kwargs):
        """Query the segment database for this flag's active segments.

        This method assumes all of the metadata for each flag have been
        filled. Minimally, the following attributes must be filled

        .. autosummary::

           ~DataQualityFlag.name
           ~DataQualityFlag.known

        Segments will be fetched from the database, with any
        :attr:`~DataQualityFlag.padding` added on-the-fly.

        This `DataQualityFlag` will be modified in-place.

        Parameters
        ----------
        source : `str`
            source of segments for this flag. This must be
            either a URL for a segment database or a path to a file on disk.

        segments : `SegmentList`, optional
            a list of valid segments during which to query, if not given,
            existing known segments for this flag will be used.

        pad : `bool`, optional, default: `True`
            apply the `~DataQualityFlag.padding` associated with this
            flag, default: `True`.

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
        tmp.populate(source=source, segments=segments, pad=pad, **kwargs)
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
        `Segment` stays within the `known` bounds.

        The :attr:`~DataQualityFlag.active` `SegmentList` is modified
        in place.

        Parameters
        ----------
        x : `float`
            number of seconds by which to protact each `Segment`.
        """
        self.active = self.active.protract(x)
        return self.active

    def pad(self, *args, **kwargs):
        """Apply a padding to each segment in this `DataQualityFlag`

        This method either takes no arguments, in which case the value of
        the :attr:`~DataQualityFlag.padding` attribute will be used,
        or two values representing the padding for the start and end of
        each segment.

        For both the `start` and `end` paddings, a positive value means
        pad forward in time, so that a positive `start` pad or negative
        `end` padding will contract a segment at one or both ends,
        and vice-versa.

        This method will apply the same padding to both the
        `~DataQualityFlag.known` and `~DataQualityFlag.active` lists,
        but will not :meth:`~DataQualityFlag.coalesce` the result.

        Parameters
        ----------
        start : `float`
            padding to apply to the start of the each segment
        end : `float`
            padding to apply to the end of each segment
        inplace : `bool`, optional, default: `False`
            modify this object in-place, default is `False`, i.e. return
            a copy of the original object with padded segments

        Returns
        -------
        paddedflag : `DataQualityFlag`
            a view of the modified flag
        """
        if len(args) == 0:
            start, end = self.padding
        elif len(args) == 2:
            start, end = args
        else:
            raise ValueError("Cannot parse (start, end) padding from %r"
                             % args)
        if kwargs.pop('inplace', False):
            new = self
        else:
            new = self.copy()
        if kwargs:
            raise TypeError("unexpected keyword argument %r"
                            % kwargs.keys()[0])
        new.known = [(s[0]+start, s[1]+end) for s in self.known]
        new.active = [(s[0]+start, s[1]+end) for s in self.active]
        return new

    def round(self):
        """Round this flag to integer segments.

        Returns
        -------
        roundedflag : `DataQualityFlag`
            A copy of the original flag with the `active` and `known` segments
            padded out to the enclosing integer boundaries.
        """
        new = self.copy()
        new.active = self._ListClass([self._EntryClass(int(floor(s[0])),
                                                       int(ceil(s[1]))) for
                                     s in new.active])
        new.known = self._ListClass([self._EntryClass(int(floor(s[0])),
                                                      int(ceil(s[1]))) for
                                     s in new.known])
        return new.coalesce()

    def coalesce(self):
        """Coalesce the segments for this flag.

        This method does two things:

        - `coalesces <SegmentList.coalesce>` the `~DataQualityFlag.known` and
          `~DataQualityFlag.active` segment lists
        - forces the `active` segments to be a proper subset of the `known`
          segments

        .. note::

            this operations is performed in-place.

        Returns
        -------
        self
            a view of this flag, not a copy.
        """
        self.known = self.known.coalesce()
        self.active = self.active.coalesce()
        self.active = (self.known & self.active).coalesce()
        return self

    def __repr__(self):
        indent = " " * len("<%s(" % self.__class__.__name__)
        known = str(self.known).replace("\n",
                                        "\n%s      " % indent).split('\n')
        if len(known) > 10:
            known = known[:3] + ['%s      ...' % indent] + known[-3:]
        active = str(self.active).replace("\n",
                                          "\n%s       " % indent).split('\n')
        if len(active) > 10:
            active = active[:3] + ['%s        ...' % indent] + active[-3:]
        return ("<{1}({2},\n{0}known={3},\n{0}active={4},\n"
                "{0}description={5})>".format(indent, self.__class__.__name__,
                                              (self.name and repr(self.name) or
                                               'No name'),
                                              '\n'.join(known),
                                              '\n'.join(active),
                                              repr(self.description)))

    def copy(self):
        """Build an exact copy of this flag.

        Returns
        -------
        flag2 : `DataQualityFlag`
            a copy of the original flag, but with a fresh memory address.
        """
        return deepcopy(self)

    def plot(self, **kwargs):
        """Plot this flag.

        Parameters
        ----------
        **kwargs
            all keyword arguments are passed to the
            :class:`~gwpy.plotter.segments.SegmentPlot` constructor.

        Returns
        -------
        plot : `~gwpy.plotter.segments.SegmentPlot`
            a new `Plot` with this flag displayed on a set of
            :class:`~gwpy.plotter.segments.SegmentAxes`.
        """
        from ..plotter import (rcParams, SegmentPlot)
        kwargs.setdefault('epoch', self.known[0][0])
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
        elif re_IFO_TAG_VERSION.match(name):
            match = re_IFO_TAG_VERSION.match(name).groupdict()
            self.ifo = match['ifo']
            self.tag = match['tag']
            self.version = int(match['version'])
        elif re_IFO_TAG.match(name):
            match = re_IFO_TAG.match(name).groupdict()
            self.ifo = match['ifo']
            self.tag = match['tag']
            self.version = None
        elif re_TAG_VERSION.match(name):
            match = re_TAG_VERSION.match(name).groupdict()
            self.ifo = None
            self.tag = match['tag']
            self.version = int(match['version'])
        else:
            raise ValueError("No flag name structure detected in '%s', flags "
                             "should be named as '{ifo}:{tag}:{version}'. "
                             "For arbitrary strings, use the "
                             "`DataQualityFlag.label` attribute" % name)
        return self.ifo, self.tag, self.version

    def __and__(self, other):
        """Find the intersection of this one and ``other``.
        """
        return self.copy().__iand__(other)

    def __iand__(self, other):
        """Intersect this flag with ``other`` in-place.
        """
        self.known &= other.known
        self.active &= other.active
        return self

    def __sub__(self, other):
        """Find the difference between this flag and another.
        """
        return self.copy().__isub__(other)

    def __isub__(self, other):
        """Subtract the ``other`` `DataQualityFlag` from this one in-place.
        """
        self.active -= other.active
        return self

    def __or__(self, other):
        """Find the union of this flag and ``other``.
        """
        return self.copy().__ior__(other)

    def __ior__(self, other):
        """Add the ``other`` `DataQualityFlag` to this one in-place.
        """
        self.known |= other.known
        self.active |= other.active
        return self

    __add__ = __or__
    __iadd__ = __ior__


class _QueryDQSegDBThread(Thread):
    """Threaded DQSegDB query
    """
    def __init__(self, inqueue, outqueue, *args, **kwargs):
        Thread.__init__(self)
        self.in_ = inqueue
        self.out = outqueue
        self.args = args
        self.kwargs = kwargs

    def run(self):
        i, flag = self.in_.get()
        self.in_.task_done()
        try:
            self.out.put(
                (i, DataQualityFlag.query_dqsegdb(flag, *self.args,
                                                  **self.kwargs)))
        except Exception as e:
            self.out.put((i, e))
        self.out.task_done()


class DataQualityDict(OrderedDict):
    """An `~collections.OrderedDict` of (key, `DataQualityFlag`) pairs.

    Since the `DataQualityDict` is an `OrderedDict`, all iterations over
    its elements retain the order in which they were inserted.
    """
    _EntryClass = DataQualityFlag

    # -----------------------------------------------------------------------
    # classmethods

    @classmethod
    def query(cls, flag, *args, **kwargs):
        """Query for segments of a set of flags.

        This method intelligently selects the `~DataQualityDict.query_segdb`
        or the `~DataQualityDict.query_dqsegdb` methods based on the
        ``url`` kwarg given.

        Parameters
        ----------
        flags : `iterable`
            A list of flag names for which to query.
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a `SegmentList`
            defining a number of summary segments
        url : `str`, optional, default: ``'https://segments.ligo.org'``
            URL of the segment database

        See Also
        --------
        DataQualityDict.query_segdb
        DataQualityDict.query_dqsegdb
            for details on the actual query engine, and documentation of
            other keyword arguments appropriate for each query

        Returns
        -------
        flag : `DataQualityFlag`
            A new `DataQualityFlag`, with the `known` and `active` lists
            filled appropriately.
        """
        url = kwargs.get('url', 'https://segments.ligo.org')
        if 'dqsegdb' in url or re.match('https://[a-z1-9-]+.ligo.org', url):
            return cls.query_dqsegdb(flag, *args, **kwargs)
        else:
            return cls.query_segdb(flag, *args, **kwargs)

    @classmethod
    def query_segdb(cls, flags, *args, **kwargs):
        """Query the inital LIGO segment database for a list of flags.

        Parameters
        ----------
        flags : `iterable`
            A list of flag names for which to query.
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a `SegmentList`
            defining a number of summary segments.
        url : `str`, optional, default: ``'https://segments.ligo.org'``
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
            qsegs = SegmentList(Segment(map(to_gps, args[0])))
        elif len(args) == 2:
            qsegs = SegmentList([Segment(map(to_gps, args))])
        else:
            raise ValueError("DataQualityDict.query_segdb must be called with "
                             "a list of flag names, and either GPS start and "
                             "stop times, or a SegmentList of query segments")
        url = kwargs.pop('url', 'https://segments.ligo.org')
        if kwargs.pop('on_error', None) is not None:
            warnings.warn("DataQualityDict.query_segdb doesn't accept the "
                          "on_error keyword argument")
        if kwargs.keys():
            raise TypeError("DataQualityDict.query_segdb has no keyword "
                            "argument '%s'" % kwargs.keys()[0])
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
                if gpsend == PosInfinity or float(gpsend) == +inf:
                    gpsend = to_gps('now').seconds
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
            if key not in out:
                out[key] = DataQualityFlag(name=flag)
            # add segments
            out[key].known.extend(summary)
            out[key].active.extend(segments)
        return out

    @classmethod
    def query_dqsegdb(cls, flags, *args, **kwargs):
        """Query the advanced LIGO DQSegDB for a list of flags.

        Parameters
        ----------
        flags : `iterable`
            A list of flag names for which to query.
        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a `SegmentList`
            defining a number of summary segments.
        on_error : `str`
            how to handle an error querying for one flag, one of

            - `'raise'` (default): raise the Exception
            - `'warn'`: print a warning
            - `'ignore'`: move onto the next flag as if nothing happened

        url : `str`, optional, default: ``'https://segments.ligo.org'``
            URL of the segment database.

        Returns
        -------
        flagdict : `DataQualityDict
            An ordered `DataQualityDict` of (name, `DataQualityFlag`)
            pairs.
        """
        # check on_error flag
        on_error = kwargs.pop('on_error', 'raise').lower()
        if on_error not in ['raise', 'warn', 'ignore']:
            raise ValueError("on_error must be one of 'raise', 'warn', "
                             "or 'ignore'")

        # set up threading
        inq = Queue()
        outq = Queue()
        for i in range(len(flags)):
            t = _QueryDQSegDBThread(inq, outq, *args, **kwargs)
            t.setDaemon(True)
            t.start()
        for i, flag in enumerate(flags):
            inq.put((i, flag))

        # capture output
        inq.join()
        outq.join()
        new = cls()
        results = zip(*sorted([outq.get() for i in range(len(flags))],
                              key=lambda x: x[0]))[1]
        for result, flag in zip(results, flags):
            if isinstance(result, Exception):
                new[flag] = cls._EntryClass(name=flag)
                result.args = ('%s [%s]' % (str(result), str(flag)),)
                if on_error == 'ignore':
                    pass
                elif on_error == 'warn':
                    warnings.warn(str(result))
                else:
                    raise result
            else:
                new[flag] = result
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
    flagdict : `DataQualityDict`
        a new `DataQualityDict` of `DataQualityFlag` entries with ``active``
        and ``known`` segments seeded from the XML tables in the given
        file.

    Notes
    -----
    """))

    @classmethod
    def from_veto_definer_file(cls, fp, start=None, end=None, ifo=None,
                               format='ligolw'):
        """Read a `DataQualityDict` from a LIGO_LW XML VetoDefinerTable.

        Parameters
        ----------
        fp : `str`
            path of veto definer file to read
        start : `~gwpy.time.LIGOTimeGPS`, `int`, optional
            GPS start time at which to restrict returned flags
        end : `~gwpy.time.LIGOTimeGPS`, `int`, optional
            GPS end time at which to restrict returned flags
        ifo : `str`, optional
            interferometer prefix whose flags you want to read
        format : `str`, optional
            format of file to read (passed to `VetoDefTable.read`),
            currently only 'ligolw' is supported

        Returns
        -------
        flags : `DataQualityDict`
            a `DataQualityDict` of flags parsed from the `veto_def_table`
            of the input file.

        Notes
        -----
        This method does not automatically `~DataQualityDict.populate`
        the `active` segment list of any flags, a separate call should
        be made for that as follows

        >>> flags = DataQualityDict.from_veto_definer_file('/path/to/file.xml')
        >>> flags.populate()

        """
        if start is not None:
            start = to_gps(start)
        if end is not None:
            end = to_gps(end)
        # read veto definer file
        from gwpy.table.lsctables import VetoDefTable
        if urlparse(fp).scheme in ['http', 'https']:
            response = request.urlopen(fp)
            local = tempfile.NamedTemporaryFile()
            with tempfile.NamedTemporaryFile() as temp:
                temp.write(response.read())
                temp.flush()
                veto_def_table = VetoDefTable.read(temp.name, format=format)
        else:
            veto_def_table = VetoDefTable.read(fp, format=format)
        # parse flag definitions
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
                out[flag.name].known.extend(flag.known)
                out[flag.name].known.coalesce()
            else:
                out[flag.name] = flag
        return out

    # -----------------------------------------------------------------------
    # instance methods

    write = writer()

    def populate(self, source='https://segments.ligo.org',
                 segments=None, pad=True, on_error='raise', **kwargs):
        """Query the segment database for each flag's active segments.

        This method assumes all of the metadata for each flag have been
        filled. Minimally, the following attributes must be filled

        .. autosummary::

           ~DataQualityFlag.name
           ~DataQualityFlag.known

        Segments will be fetched from the database, with any
        :attr:`~DataQualityFlag.padding` added on-the-fly.

        Entries in this dict will be modified in-place.

        Parameters
        ----------
        source : `str`
            source of segments for this flag. This must be
            either a URL for a segment database or a path to a file on disk.

        segments : `SegmentList`, optional
            a list of known segments during which to query, if not given,
            existing known segments for flags will be used.

        pad : `bool`, optional, default: `True`
            apply the `~DataQualityFlag.padding` associated with each
            flag, default: `True`.

        on_error : `str`
            how to handle an error querying for one flag, one of

            - `'raise'` (default): raise the Exception
            - `'warn'`: print a warning
            - `'ignore'`: move onto the next flag as if nothing happened

        **kwargs
            any other keyword arguments to be passed to
            :meth:`DataQualityFlag.query` or :meth:`DataQualityFlag.read`.

        Returns
        -------
        self : `DataQualityDict`
            a reference to the modified DataQualityDict
        """
        # check on_error flag
        if on_error not in ['raise', 'warn', 'ignore']:
            raise ValueError("on_error must be one of 'raise', 'warn', "
                             "or 'ignore'")
        # format source
        source = urlparse(source)
        # perform query for all segments
        if source.netloc and segments is not None:
            segments = SegmentList(map(Segment, segments))
            tmp = type(self).query(self.keys(), segments, url=source.geturl(),
                                   on_error=on_error, **kwargs)
        elif not source.netloc:
            tmp = type(self).read(source.geturl(), self.name, **kwargs)
        # apply padding and wrap to given known segments
        for key in self:
            if segments is None and source.netloc:
                try:
                    tmp = {key: self[key].query(
                        self[key].name, self[key].known, **kwargs)}
                except URLError as e:
                    if on_error == 'ignore':
                        pass
                    elif on_error == 'warn':
                        warnings.warn('Error querying for %s: %s' % (key, e))
                    else:
                        raise
                    continue
            self[key].known &= tmp[key].known
            self[key].active = tmp[key].active
            if pad:
                self[key] = self[key].pad(inplace=True)
                if segments is not None:
                    self[key].known &= segments
                    self[key].active &= segments
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
        """Return the union of all flags in this dict

        Returns
        -------
        union : `DataQualityFlag`
            a new `DataQualityFlag` who's active and known segments
            are the union of those of the values of this dict
        """
        usegs = reduce(operator.or_, self.itervalues())
        usegs.name = ' | '.join(self.iterkeys())
        return usegs

    def intersection(self):
        """Return the intersection of all flags in this dict

        Returns
        -------
        intersection : `DataQualityFlag`
            a new `DataQualityFlag` who's active and known segments
            are the intersection of those of the values of this dict
        """
        isegs = reduce(operator.and_, self.itervalues())
        isegs.name = ' & '.join(self.iterkeys())
        return isegs

    def plot(self, label='key', **kwargs):
        """Plot the data for this dict.

        Parameters
        ----------
        label : `str`, optional
            labelling system to use, or fixed label for all flags,
            special values include

            - ``'key'``: use the key of the `DataQualityDict`,
            - ``'name'``: use the :attr:`~DataQualityFlag.name` of the flag

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
