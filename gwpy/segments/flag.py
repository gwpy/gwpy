# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2021)
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

import datetime
import json
import operator
import os
import re
import warnings
from io import BytesIO
from collections import OrderedDict
from copy import (copy as shallowcopy, deepcopy)
from functools import reduce
from math import (floor, ceil)
from queue import Queue
from threading import Thread
from urllib.error import (URLError, HTTPError)
from urllib.parse import urlparse

from numpy import inf

from astropy.io import registry as io_registry
from astropy.utils.data import get_readable_fileobj

from gwosc import timeline

from dqsegdb2.query import query_segments

from ..io.mp import read_multi as io_read_multi
from ..time import to_gps, LIGOTimeGPS
from ..utils.misc import if_not_none
from .segments import Segment, SegmentList

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['DataQualityFlag', 'DataQualityDict']

re_IFO_TAG_VERSION = re.compile(
    r"\A(?P<ifo>[A-Z]\d):(?P<tag>[^/]+):(?P<version>\d+)\Z")
re_IFO_TAG = re.compile(r"\A(?P<ifo>[A-Z]\d):(?P<tag>[^/]+)\Z")
re_TAG_VERSION = re.compile(r"\A(?P<tag>[^/]+):(?P<version>\d+)\Z")

DEFAULT_SEGMENT_SERVER = os.getenv('DEFAULT_SEGMENT_SERVER',
                                   'https://segments.ligo.org')


# -- utilities ----------------------------------------------------------------

def _parse_query_segments(args, func):
    """Parse *args for query_dqsegdb() or query_segdb()

    Returns a SegmentList in all cases
    """
    # user passed SegmentList
    if len(args) == 1 and isinstance(args[0], SegmentList):
        return args[0]

    # otherwise unpack two arguments as a segment
    if len(args) == 1:
        args = args[0]

    # if not two arguments, panic
    try:
        start, end = args
    except ValueError as exc:
        exc.args = ('{0}() takes 2 arguments for start and end GPS time, '
                    'or 1 argument containing a Segment or SegmentList'.format(
                        func.__name__),)
        raise

    # return list with one Segment
    return SegmentList([Segment(to_gps(start), to_gps(end))])


# -- DataQualityFlag ----------------------------------------------------------

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
                 category=None, description=None, isgood=True, padding=None):
        """Define a new DataQualityFlag.
        """
        self.name = name
        self.known = known
        self.active = active
        self.label = label
        self.category = category
        self.description = description
        self.isgood = isgood
        self.padding = padding

    # -- properties -----------------------------

    @property
    def name(self):
        """The name associated with this flag.

        This normally takes the form {ifo}:{tag}:{version}. If found,
        each component is stored separately the associated attributes.

        :type: `str`
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        try:
            self._parse_name(name)
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
        self._version = int(v) if v is not None else None

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
    def category(self):
        """Veto category for this flag.

        :type: `int`
        """
        return self._category

    @category.setter
    def category(self, cat):
        self._category = if_not_none(int, cat)

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
        return self._padding

    @padding.setter
    def padding(self, pad):
        if pad is None:
            pad = (None, None)
        self._padding = tuple(float(p or 0.) for p in pad)

    @padding.deleter
    def padding(self):
        self._padding = (0., 0.)

    # -- read-only properties -------------------

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

    # -- classmethods ---------------------------

    @classmethod
    def query_dqsegdb(cls, flag, *args, **kwargs):
        """Query the advanced LIGO DQSegDB for the given flag

        Parameters
        ----------
        flag : `str`
            The name of the flag for which to query

        *args
            Either, two `float`-like numbers indicating the
            GPS [start, stop) interval, or a `SegmentList`
            defining a number of summary segments

        url : `str`, optional
            URL of the segment database, defaults to
            ``$DEFAULT_SEGMENT_SERVER`` environment variable, or
            ``'https://segments.ligo.org'``

        Returns
        -------
        flag : `DataQualityFlag`
            A new `DataQualityFlag`, with the `known` and `active` lists
            filled appropriately.
        """
        # parse arguments
        qsegs = _parse_query_segments(args, cls.query_dqsegdb)

        # get server
        url = kwargs.pop('url', DEFAULT_SEGMENT_SERVER)

        # parse flag
        out = cls(name=flag)
        if out.ifo is None or out.tag is None:
            raise ValueError("Cannot parse ifo or tag (name) for flag %r"
                             % flag)

        # process query
        for start, end in qsegs:
            # handle infinities
            if float(end) == +inf:
                end = int(to_gps('now'))

            # query
            try:
                data = query_segments(flag, int(start), int(end), host=url)
            except HTTPError as exc:
                if exc.code == 404:  # if not found, annotate flag name
                    exc.msg += ' [{0}]'.format(flag)
                raise

            # read from json buffer
            new = cls.read(
                BytesIO(json.dumps(data).encode('utf-8')),
                format='json',
            )

            # restrict to query segments
            segl = SegmentList([Segment(start, end)])
            new.known &= segl
            new.active &= segl
            out += new
            # replace metadata
            out.description = new.description
            out.isgood = new.isgood

        return out

    # alias for compatibility
    query = query_dqsegdb

    @classmethod
    def fetch_open_data(cls, flag, start, end, **kwargs):
        """Fetch Open Data timeline segments into a flag.

        flag : `str`
            the name of the flag to query

        start : `int`, `str`
            the GPS start time (or parseable date string) to query

        end : `int`, `str`
            the GPS end time (or parseable date string) to query

        verbose : `bool`, optional
            show verbose download progress, default: `False`

        timeout : `int`, optional
            timeout for download (seconds)

        host : `str`, optional
            URL of GWOSC host, default: ``'https://gw-openscience.org'``

        Returns
        -------
        flag : `DataQualityFlag`
            a new flag with `active` segments filled from Open Data

        Examples
        --------
        >>> from gwpy.segments import DataQualityFlag
        >>> print(DataQualityFlag.fetch_open_data('H1_DATA', 'Jan 1 2010',
        ...                                       'Jan 2 2010'))
        <DataQualityFlag('H1:DATA',
                         known=[[946339215 ... 946425615)],
                         active=[[946340946 ... 946351800)
                                 [946356479 ... 946360620)
                                 [946362652 ... 946369150)
                                 [946372854 ... 946382630)
                                 [946395595 ... 946396751)
                                 [946400173 ... 946404977)
                                 [946412312 ... 946413577)
                                 [946415770 ... 946422986)],
                         description=None)>
        """
        start = to_gps(start).gpsSeconds
        end = to_gps(end).gpsSeconds
        known = [(start, end)]
        active = timeline.get_segments(flag, start, end, **kwargs)
        return cls(flag.replace('_', ':', 1), known=known, active=active,
                   label=flag)

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read segments from file into a `DataQualityFlag`.

        Parameters
        ----------
        filename : `str`
            path of file to read

        name : `str`, optional
            name of flag to read from file, otherwise read all segments.

        format : `str`, optional
            source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        coltype : `type`, optional, default: `float`
            datatype to force for segment times, only valid for
            ``format='segwizard'``.

        strict : `bool`, optional, default: `True`
            require segment start and stop times match printed duration,
            only valid for ``format='segwizard'``.

        coalesce : `bool`, optional
            if `True` coalesce the all segment lists before returning,
            otherwise return exactly as contained in file(s).

        nproc : `int`, optional, default: 1
            number of CPUs to use for parallel reading of multiple files

        verbose : `bool`, optional, default: `False`
            print a progress bar showing read status

        Returns
        -------
        dqflag : `DataQualityFlag`
            formatted `DataQualityFlag` containing the active and known
            segments read from file.

        Raises
        ------
        IndexError
            if ``source`` is an empty list

        Notes
        -----"""
        if 'flag' in kwargs:  # pragma: no cover
            warnings.warn('\'flag\' keyword was renamed \'name\', this '
                          'warning will result in an error in the future')
            kwargs.setdefault('name', kwargs.pop('flags'))
        coalesce = kwargs.pop('coalesce', False)

        def combiner(flags):
            """Combine `DataQualityFlag` from each file into a single object
            """
            out = flags[0]
            for flag in flags[1:]:
                out.known += flag.known
                out.active += flag.active
            if coalesce:
                return out.coalesce()
            return out

        return io_read_multi(combiner, cls, source, *args, **kwargs)

    @classmethod
    def from_veto_def(cls, veto):
        """Define a `DataQualityFlag` from a `VetoDef`

        Parameters
        ----------
        veto : :class:`~ligo.lw.lsctables.VetoDef`
            veto definition to convert from
        """
        name = '%s:%s' % (veto.ifo, veto.name)
        try:
            name += ':%d' % int(veto.version)
        except TypeError:
            pass
        known = Segment(veto.start_time, veto.end_time or +inf)
        pad = (veto.start_pad, veto.end_pad)
        return cls(name=name, known=[known], category=veto.category,
                   description=veto.comment, padding=pad)

    # -- methods --------------------------------

    def write(self, target, *args, **kwargs):
        """Write this `DataQualityFlag` to file

        Notes
        -----"""
        return io_registry.write(self, target, *args, **kwargs)

    def populate(self, source=DEFAULT_SEGMENT_SERVER, segments=None,
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
            a list of segments during which to query, if not given,
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
        if not args:
            start, end = self.padding
        else:
            start, end = args

        if kwargs.pop('inplace', False):
            new = self
        else:
            new = self.copy()
        if kwargs:
            raise TypeError("unexpected keyword argument %r"
                            % list(kwargs.keys())[0])
        new.known = [(s[0]+start, s[1]+end) for s in self.known]
        new.active = [(s[0]+start, s[1]+end) for s in self.active]
        return new

    def round(self, contract=False):
        """Round this flag to integer segments.

        Parameters
        ----------
        contract : `bool`, optional
            if `False` (default) expand each segment to the containing
            integer boundaries, otherwise contract each segment to the
            contained boundaries

        Returns
        -------
        roundedflag : `DataQualityFlag`
            A copy of the original flag with the `active` and `known` segments
            padded out to integer boundaries.
        """
        def _round(seg):
            if contract:  # round inwards
                a = type(seg[0])(ceil(seg[0]))
                b = type(seg[1])(floor(seg[1]))
            else:  # round outwards
                a = type(seg[0])(floor(seg[0]))
                b = type(seg[1])(ceil(seg[1]))
            if a >= b:  # if segment is too short, return 'null' segment
                return type(seg)(0, 0)  # will get coalesced away
            return type(seg)(a, b)

        new = self.copy()
        new.active = type(new.active)(map(_round, new.active))
        new.known = type(new.known)(map(_round, new.known))
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
        prefix = "<{}(".format(type(self).__name__)
        suffix = ")>"
        indent = " " * len(prefix)

        # format segment lists
        known = str(self.known).replace(
            "\n",
            "\n{}      ".format(indent),
        ).split("\n")
        if len(known) > 10:  # use ellipsis
            known = known[:3] + ['{}      ...'.format(indent)] + known[-3:]
        active = str(self.active).replace(
            "\n",
            "\n{}       ".format(indent),
        ).split('\n')
        if len(active) > 10:  # use ellipsis
            active = active[:3] + ['{}      ...'.format(indent)] + active[-3:]

        # print the thing
        return "".join((
            prefix,
            "\n{}".format(indent).join([
                "{},".format(repr(self.name)),
                "known={}".format("\n".join(known)),
                "active={}".format("\n".join(active)),
                "description={}".format(repr(self.description)),
            ]),
            suffix,
        ))

    def copy(self):
        """Build an exact copy of this flag.

        Returns
        -------
        flag2 : `DataQualityFlag`
            a copy of the original flag, but with a fresh memory address.
        """
        return deepcopy(self)

    def plot(self, figsize=(12, 4), xscale='auto-gps', **kwargs):
        """Plot this flag on a segments projection.

        Parameters
        ----------
        **kwargs
            all keyword arguments are passed to the
            :class:`~gwpy.plot.Plot` constructor.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            the newly created figure, with populated Axes.

        See also
        --------
        matplotlib.pyplot.figure
            for documentation of keyword arguments used to create the
            figure
        matplotlib.figure.Figure.add_subplot
            for documentation of keyword arguments used to create the
            axes
        gwpy.plot.SegmentAxes.plot_segmentlist
            for documentation of keyword arguments used in rendering the data
        """
        from matplotlib import rcParams
        from ..plot import Plot

        if self.label:
            kwargs.setdefault('label', self.label)
        elif rcParams['text.usetex']:
            kwargs.setdefault('label', self.texname)
        else:
            kwargs.setdefault('label', self.name)

        kwargs.update(figsize=figsize, xscale=xscale)
        return Plot(self, projection='segments', **kwargs)

    def _parse_name(self, name):
        """Internal method to parse a `string` name into constituent
        `ifo, `name` and `version` components.

        Parameters
        ----------
        name : `str`, `None`
            the full name of a `DataQualityFlag` to parse, e.g.
            ``'H1:DMT-SCIENCE:1'``, or `None` to set all components
            to `None`

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
        if isinstance(name, bytes):
            name = name.decode('utf-8')
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
        self.known &= other.known
        self.active -= other.active
        self.active &= self.known
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

    def __xor__(self, other):
        """Find the exclusive OR of this one and ``other``.
        """
        return self.copy().__ixor__(other)

    def __ixor__(self, other):
        """Exclusive OR this flag with ``other`` in-place.
        """
        self.known &= other.known
        self.active ^= other.active
        return self

    def __invert__(self):
        new = self.copy()
        new.active = ~self.active
        new.active &= new.known
        return new


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
        except Exception as exc:
            self.out.put((i, exc))
        self.out.task_done()


class DataQualityDict(OrderedDict):
    """An `~collections.OrderedDict` of (key, `DataQualityFlag`) pairs.

    Since the `DataQualityDict` is an `OrderedDict`, all iterations over
    its elements retain the order in which they were inserted.
    """
    _EntryClass = DataQualityFlag

    # -- classmethods ---------------------------

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

        url : `str`, optional
            URL of the segment database, defaults to
            ``$DEFAULT_SEGMENT_SERVER`` environment variable, or
            ``'https://segments.ligo.org'``

        Returns
        -------
        flagdict : `DataQualityDict`
            An ordered `DataQualityDict` of (name, `DataQualityFlag`)
            pairs.
        """
        # check on_error flag
        on_error = kwargs.pop('on_error', 'raise').lower()
        if on_error not in ['raise', 'warn', 'ignore']:
            raise ValueError("on_error must be one of 'raise', 'warn', "
                             "or 'ignore'")

        # parse segments
        qsegs = _parse_query_segments(args, cls.query_dqsegdb)

        # set up threading
        inq = Queue()
        outq = Queue()
        for i in range(len(flags)):
            t = _QueryDQSegDBThread(inq, outq, qsegs, **kwargs)
            t.daemon = True
            t.start()
        for i, flag in enumerate(flags):
            inq.put((i, flag))

        # capture output
        inq.join()
        outq.join()
        new = cls()
        results = list(zip(*sorted([outq.get() for i in range(len(flags))],
                                   key=lambda x: x[0])))[1]
        for result, flag in zip(results, flags):
            if isinstance(result, Exception):
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

    # alias for compatibility
    query = query_dqsegdb

    @classmethod
    def read(cls, source, names=None, format=None, **kwargs):
        """Read segments from file into a `DataQualityDict`

        Parameters
        ----------
        source : `str`
            path of file to read

        format : `str`, optional
            source format identifier. If not given, the format will be
            detected if possible. See below for list of acceptable
            formats.

        names : `list`, optional, default: read all names found
            list of names to read, by default all names are read separately.

        coalesce : `bool`, optional
            if `True` coalesce the all segment lists before returning,
            otherwise return exactly as contained in file(s).

        nproc : `int`, optional, default: 1
            number of CPUs to use for parallel reading of multiple files

        verbose : `bool`, optional, default: `False`
            print a progress bar showing read status

        Returns
        -------
        flagdict : `DataQualityDict`
            a new `DataQualityDict` of `DataQualityFlag` entries with
            ``active`` and ``known`` segments seeded from the XML tables
            in the given file.

        Notes
        -----"""
        on_missing = kwargs.pop('on_missing', 'error')
        coalesce = kwargs.pop('coalesce', False)

        if 'flags' in kwargs:  # pragma: no cover
            warnings.warn('\'flags\' keyword was renamed \'names\', this '
                          'warning will result in an error in the future')
            names = kwargs.pop('flags')

        def combiner(inputs):
            out = cls()

            # check all names are contained
            required = set(names or [])
            found = set(name for dqdict in inputs for name in dqdict)
            for name in required - found:  # validate all names are found once
                msg = '{!r} not found in any input file'.format(name)
                if on_missing == 'ignore':
                    continue
                if on_missing == 'warn':
                    warnings.warn(msg)
                else:
                    raise ValueError(msg)

            # combine flags
            for dqdict in inputs:
                for flag in dqdict:
                    try:  # repeated occurence
                        out[flag].known.extend(dqdict[flag].known)
                        out[flag].active.extend(dqdict[flag].active)
                    except KeyError:  # first occurence
                        out[flag] = dqdict[flag]
            if coalesce:
                return out.coalesce()
            return out

        return io_read_multi(combiner, cls, source, names=names,
                             format=format, on_missing='ignore', **kwargs)

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
            format of file to read, currently only 'ligolw' is supported

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
        if format != 'ligolw':
            raise NotImplementedError("Reading veto definer from non-ligolw "
                                      "format file is not currently "
                                      "supported")

        # read veto definer file
        with get_readable_fileobj(fp, show_progress=False) as fobj:
            from ..io.ligolw import read_table as read_ligolw_table
            veto_def_table = read_ligolw_table(fobj, 'veto_definer')

        if start is not None:
            start = to_gps(start)
        if end is not None:
            end = to_gps(end)

        # parse flag definitions
        out = cls()
        for row in veto_def_table:
            if ifo and row.ifo != ifo:
                continue
            if start and 0 < row.end_time <= start:
                continue
            elif start:
                row.start_time = max(row.start_time, start)
            if end and row.start_time >= end:
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

    @classmethod
    def from_ligolw_tables(cls, segmentdeftable, segmentsumtable,
                           segmenttable, names=None, gpstype=LIGOTimeGPS,
                           on_missing='error'):
        """Build a `DataQualityDict` from a set of LIGO_LW segment tables

        Parameters
        ----------
        segmentdeftable : :class:`~ligo.lw.lsctables.SegmentDefTable`
            the ``segment_definer`` table to read

        segmentsumtable : :class:`~ligo.lw.lsctables.SegmentSumTable`
            the ``segment_summary`` table to read

        segmenttable : :class:`~ligo.lw.lsctables.SegmentTable`
            the ``segment`` table to read

        names : `list` of `str`, optional
            a list of flag names to read, defaults to returning all

        gpstype : `type`, `callable`, optional
            class to use for GPS times in returned objects, can be a function
            to convert GPS time to something else, default is
            `~gwpy.time.LIGOTimeGPS`

        on_missing : `str`, optional
            action to take when a one or more ``names`` are not found in
            the ``segment_definer`` table, one of

            - ``'ignore'`` : do nothing
            - ``'warn'`` : print a warning
            - ``error'`` : raise a `ValueError`

        Returns
        -------
        dqdict : `DataQualityDict`
            a dict of `DataQualityFlag` objects populated from the LIGO_LW
            tables
        """
        out = cls()

        id_ = dict()  # need to record relative IDs from LIGO_LW

        # read segment definers and generate DataQualityFlag object
        for row in segmentdeftable:
            ifos = sorted(row.instruments)
            ifo = ''.join(ifos) if ifos else None
            tag = row.name
            version = row.version
            name = ':'.join([str(k) for k in (ifo, tag, version) if
                             k is not None])
            if names is None or name in names:
                out[name] = DataQualityFlag(name)
                thisid = int(row.segment_def_id)
                try:
                    id_[name].append(thisid)
                except (AttributeError, KeyError):
                    id_[name] = [thisid]

        # verify all requested flags were found
        for flag in names or []:
            if flag not in out and on_missing != 'ignore':
                msg = ("no segment definition found for flag={0!r} in "
                       "file".format(flag))
                if on_missing == 'warn':
                    warnings.warn(msg)
                else:
                    raise ValueError(msg)

        # parse a table into the target DataQualityDict
        def _parse_segments(table, listattr):
            for row in table:
                for flag in out:
                    # match row ID to list of IDs found for this flag
                    if int(row.segment_def_id) in id_[flag]:
                        getattr(out[flag], listattr).append(
                            Segment(*map(gpstype, row.segment)),
                        )
                        break

        # read segment summary table as 'known'
        _parse_segments(segmentsumtable, "known")

        # read segment table as 'active'
        _parse_segments(segmenttable, "active")

        return out

    def to_ligolw_tables(self, **attrs):
        """Convert this `DataQualityDict` into a trio of LIGO_LW segment tables

        Parameters
        ----------
        **attrs
            other attributes to add to all rows in all tables
            (e.g. ``'process_id'``)

        Returns
        -------
        segmentdeftable : :class:`~ligo.lw.lsctables.SegmentDefTable`
            the ``segment_definer`` table

        segmentsumtable : :class:`~ligo.lw.lsctables.SegmentSumTable`
            the ``segment_summary`` table

        segmenttable : :class:`~ligo.lw.lsctables.SegmentTable`
            the ``segment`` table
        """
        from ligo.lw import lsctables
        from ..io.ligolw import to_table_type as to_ligolw_table_type

        SegmentDefTable = lsctables.SegmentDefTable
        SegmentSumTable = lsctables.SegmentSumTable
        SegmentTable = lsctables.SegmentTable
        segdeftab = lsctables.New(SegmentDefTable)
        segsumtab = lsctables.New(SegmentSumTable)
        segtab = lsctables.New(SegmentTable)

        def _write_attrs(table, row):
            for key, val in attrs.items():
                setattr(row, key, to_ligolw_table_type(val, table, key))

        # write flags to tables
        for flag in self.values():
            # segment definer
            segdef = segdeftab.RowType()
            for col in segdeftab.columnnames:  # default all columns to None
                setattr(segdef, col, None)
            segdef.instruments = {flag.ifo}
            segdef.name = flag.tag
            segdef.version = flag.version
            segdef.comment = flag.description
            segdef.insertion_time = to_gps(datetime.datetime.now()).gpsSeconds
            segdef.segment_def_id = SegmentDefTable.get_next_id()
            _write_attrs(segdeftab, segdef)
            segdeftab.append(segdef)

            # write segment summary (known segments)
            for vseg in flag.known:
                segsum = segsumtab.RowType()
                for col in segsumtab.columnnames:  # default columns to None
                    setattr(segsum, col, None)
                segsum.segment_def_id = segdef.segment_def_id
                segsum.segment = map(LIGOTimeGPS, vseg)
                segsum.comment = None
                segsum.segment_sum_id = SegmentSumTable.get_next_id()
                _write_attrs(segsumtab, segsum)
                segsumtab.append(segsum)

            # write segment table (active segments)
            for aseg in flag.active:
                seg = segtab.RowType()
                for col in segtab.columnnames:  # default all columns to None
                    setattr(seg, col, None)
                seg.segment_def_id = segdef.segment_def_id
                seg.segment = map(LIGOTimeGPS, aseg)
                seg.segment_id = SegmentTable.get_next_id()
                _write_attrs(segtab, seg)
                segtab.append(seg)

        return segdeftab, segsumtab, segtab

    # -- methods --------------------------------

    def write(self, target, *args, **kwargs):
        """Write this `DataQualityDict` to file

        Notes
        -----"""
        return io_registry.write(self, target, *args, **kwargs)

    def coalesce(self):
        """Coalesce all segments lists in this `DataQualityDict`.

        **This method modifies this object in-place.**

        Returns
        -------
        self
            a view of this flag, not a copy.
        """
        for flag in self:
            self[flag].coalesce()
        return self

    def populate(self, source=DEFAULT_SEGMENT_SERVER,
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
            tmp = type(self).read(source.geturl(), **kwargs)
        # apply padding and wrap to given known segments
        for key in self:
            if segments is None and source.netloc:
                try:
                    tmp = {key: self[key].query(
                        self[key].name, self[key].known, **kwargs)}
                except URLError as exc:
                    if on_error == 'ignore':
                        pass
                    elif on_error == 'warn':
                        warnings.warn('Error querying for %s: %s' % (key, exc))
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

    def copy(self, deep=False):
        """Build a copy of this dictionary.

        Parameters
        ----------
        deep : `bool`, optional, default: `False`
            perform a deep copy of the original dictionary with a fresh
            memory address

        Returns
        -------
        flag2 : `DataQualityFlag`
            a copy of the original dictionary
        """
        if deep:
            return deepcopy(self)
        return super().copy()

    def __iand__(self, other):
        for key, value in other.items():
            if key in self:
                self[key] &= value
            else:
                self[key] = self._EntryClass()
        return self

    def __and__(self, other):
        if (
            sum(len(s.active) for s in self.values())
            <= sum(len(s.active) for s in other.values())
        ):
            return self.copy(deep=True).__iand__(other)
        return other.copy(deep=True).__iand__(self)

    def __ior__(self, other):
        for key, value in other.items():
            if key in self:
                self[key] |= value
            else:
                self[key] = shallowcopy(value)
        return self

    def __or__(self, other):
        if (
            sum(len(s.active) for s in self.values())
            >= sum(len(s.active) for s in other.values())
        ):
            return self.copy(deep=True).__ior__(other)
        return other.copy(deep=True).__ior__(self)

    __iadd__ = __ior__
    __add__ = __or__

    def __isub__(self, other):
        for key, value in other.items():
            if key in self:
                self[key] -= value
        return self

    def __sub__(self, other):
        return self.copy(deep=True).__isub__(other)

    def __ixor__(self, other):
        for key, value in other.items():
            if key in self:
                self[key] ^= value
        return self

    def __xor__(self, other):
        return self.copy(deep=True).__ixor__(other)

    def __invert__(self):
        new = self.copy(deep=True)
        for key, value in new.items():
            new[key] = ~value
        return new

    def union(self):
        """Return the union of all flags in this dict

        Returns
        -------
        union : `DataQualityFlag`
            a new `DataQualityFlag` who's active and known segments
            are the union of those of the values of this dict
        """
        usegs = reduce(operator.or_, self.values())
        usegs.name = ' | '.join(self.keys())
        return usegs

    def intersection(self):
        """Return the intersection of all flags in this dict

        Returns
        -------
        intersection : `DataQualityFlag`
            a new `DataQualityFlag` who's active and known segments
            are the intersection of those of the values of this dict
        """
        isegs = reduce(operator.and_, self.values())
        isegs.name = ' & '.join(self.keys())
        return isegs

    def plot(self, label='key', **kwargs):
        """Plot this flag on a segments projection.

        Parameters
        ----------
        label : `str`, optional
            Labelling system to use, or fixed label for all flags,
            special values include

            - ``'key'``: use the key of the `DataQualityDict`,
            - ``'name'``: use the :attr:`~DataQualityFlag.name` of the flag

            If anything else, that fixed label will be used for all lines.

        **kwargs
            all keyword arguments are passed to the
            :class:`~gwpy.plot.Plot` constructor.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            the newly created figure, with populated Axes.

        See also
        --------
        matplotlib.pyplot.figure
            for documentation of keyword arguments used to create the
            figure
        matplotlib.figure.Figure.add_subplot
            for documentation of keyword arguments used to create the
            axes
        gwpy.plot.SegmentAxes.plot_segmentlist
            for documentation of keyword arguments used in rendering the data
        """
        # make plot
        from ..plot import Plot
        plot = Plot(self, projection='segments', **kwargs)

        # update labels
        artists = [x for ax in plot.axes for x in ax.collections]
        for key, artist in zip(self, artists):
            if label.lower() == 'name':
                lab = self[key].name
            elif label.lower() != 'key':
                lab = key
            else:
                lab = label
            artist.set_label(lab)

        return plot
