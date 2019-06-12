# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

"""Utilities for auto-discovery of GW data files.

Automatic discovery of file paths for both LIGO and Virgo index solutions
(``gwdatafind`` or ``ffl``, respectvely) is supported.

The functions in this module are highly reliant on having local access to
files (either directly, or via NFS, or CVMFS).

Data discovery using the DataFind service requires the `gwdatafind` Python
package, and either the ``LIGO_DATAFIND_SERVER`` environment variable to be
set, or the ``host`` keyword must be passed to :func:`find_urls` and friends.

Data discovery using the Virgo FFL system requires the ``FFLPATH`` environment
variable to point to the directory containing FFL files, **or** the
``VIRGODATA`` environment variable to point to a directory containing an
``ffl` subdirectory, which contains FFL files.
"""

import os
import os.path
import re
import warnings
from functools import wraps

from six.moves.http_client import HTTPException
from six.moves.urllib.parse import urlparse

from ligo.segments import (
    segment as LigoSegment,
    segmentlist as LigoSegmentList,
)
from ..time import to_gps
from .cache import (cache_segments, read_cache_entry, _iter_cache)
from .gwf import (num_channels, iter_channel_names)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# special-case frame types
SECOND_TREND_TYPE = re.compile(r'\A(.*_)?T\Z')  # T or anything ending in _T
MINUTE_TREND_TYPE = re.compile(r'\A(.*_)?M\Z')  # M or anything ending in _M
GRB_TYPE = re.compile(r'^(?!.*_GRB\d{6}([A-Z])?$)')
HIGH_PRIORITY_TYPE = re.compile(
    r'\A[A-Z]\d_HOFT_C\d\d(_T\d{7}_v\d)?\Z'  # X1_HOFT_CXY
)
LOW_PRIORITY_TYPE = re.compile(
    r'(_GRB\d{6}([A-Z])?\Z|'  # endswith _GRBYYMMDD{A}
    r'_bck\Z|'  # endswith _bck
    r'\AT1200307_V4_EARLY_RECOLORED_V2\Z)'  # annoying recoloured HOFT type
)


# -- utilities ----------------------------------------------------------------


class FflConnection(object):
    """API for Virgo FFL queries that mimics `gwdatafind.http.HTTPConnection`
    """
    _EXTENSION = 'ffl'
    _SITE_REGEX = re.compile(r'\A(\w+)-')

    def __init__(self, ffldir=None):
        self.ffldir = ffldir or self._get_ffl_dir()
        self.paths = {}
        self.cache = {}
        self._find_paths()

    # -- utilities ------------------------------

    @staticmethod
    def _get_ffl_dir():
        if 'FFLPATH' in os.environ:
            return os.environ['FFLPATH']
        if 'VIRGODATA' in os.environ:
            return os.path.join(os.environ['VIRGODATA'], 'ffl')
        raise KeyError("failed to parse FFTPATH from environment, please set "
                       "FFLPATH to point to the directory containing FFL "
                       "files")

    @classmethod
    def _is_ffl_file(cls, path):
        return path.endswith('.{0}'.format(cls._EXTENSION))

    def _find_paths(self):
        _is_ffl = self._is_ffl_file

        # reset
        paths = self.paths = {}

        # scan directory tree
        for root, _, files in os.walk(self.ffldir):
            for name in filter(_is_ffl, files):
                path = os.path.join(root, name)
                try:
                    site, tag = self._get_site_tag(path)
                except (OSError, IOError, AttributeError):
                    # OSError: file is empty (or cannot be read at all)
                    # IOError: as above on python2
                    # AttributeError: last entry didn't match _SITE_REGEX
                    continue
                paths[(site, tag)] = path

    # -- readers --------------------------------

    def _read_ffl_cache(self, site, tag):
        key = (site, tag)
        path = self.ffl_path(site, tag)

        # use cached result if already read, and file not modified since
        try:
            mtime = self.cache[key][0]
        except KeyError:
            mtime = 0
        newm = os.path.getmtime(path)
        if newm > mtime:  # read FFL file
            def _update_metadata(entry):
                return type(entry)(site, tag, entry.segment, entry.path)
            with open(path, 'r') as fobj:
                cache = list(map(_update_metadata, _iter_cache(fobj)))
            self.cache[key] = newm, cache
        return self.cache[key][-1]

    @staticmethod
    def _read_last_line(path):
        with open(path, 'rb') as fobj:
            # read last line of file only
            fobj.seek(-2, os.SEEK_END)
            while fobj.read(1) != b"\n":
                fobj.seek(-2, os.SEEK_CUR)
            line = fobj.readline().rstrip()
            if isinstance(line, bytes):
                return line.decode('utf-8')
            return line

    def _get_site_tag(self, path):
        # tag is just name of file minus extension
        tag = os.path.splitext(os.path.basename(path))[0]

        # need to read first file from FFL to get site (IFO)
        last = self._read_last_line(path).split()[0]
        site = self._SITE_REGEX.match(os.path.basename(last)).groups()[0]

        return site, tag

    # -- UI -------------------------------------

    def ffl_path(self, site, frametype):
        """Returns the path of the FFL file for the given site and frametype

        Examples
        --------
        >>> from gwpy.io.datafind import FflConnection
        >>> conn = FflConnection()
        >>> print(conn.ffl_path('V', 'V1Online'))
        /virgoData/ffl/V1Online.ffl
        """
        try:
            return self.paths[(site, frametype)]
        except KeyError:
            self._find_paths()
            return self.paths[(site, frametype)]

    def find_types(self, site=None, match=r'^(?!lastfile|spectro|\.).*'):
        """Return the list of known data types.

        This is just the basename of each FFL file found in the
        FFL directory (minus the ``.ffl`` extension)
        """
        self._find_paths()
        types = [tag for (site_, tag) in self.paths if site in (None, site_)]
        if match is not None:
            match = re.compile(match)
            return list(filter(match.search, types))
        return types

    def find_urls(self, site, frametype, gpsstart, gpsend,
                  match=None, on_gaps='warn'):
        """Find all files of the given type in the [start, end) GPS interval.
        """
        span = LigoSegment(gpsstart, gpsend)
        cache = [e for e in self._read_ffl_cache(site, frametype) if
                 e.observatory == site and e.description == frametype and
                 e.segment.intersects(span)]
        urls = [e.path for e in cache]
        missing = LigoSegmentList([span]) - cache_segments(cache)

        if match:
            match = re.compile(match)
            urls = list(filter(match.search, urls))

        # no missing data or don't care, return
        if on_gaps == 'ignore' or not missing:
            return urls

        # handle missing data
        msg = 'Missing segments: \n{0}'.format('\n'.join(map(str, missing)))
        if on_gaps == 'warn':
            warnings.warn(msg)
            return urls
        raise RuntimeError(msg)

    def find_latest(self, site, frametype, on_missing='warn'):
        """Return the most recent file of a given type.
        """
        try:
            urls = [self.cache[(site, frametype)][-1].path]
        except KeyError:
            try:
                path = self.ffl_path(site, frametype)
                urls = [read_cache_entry(self._read_last_line(path))]
            except (KeyError, OSError):
                urls = []
        if urls or on_missing == 'ignore':
            return urls

        # handle no files
        msg = 'No files found'
        if on_missing == 'warn':
            warnings.warn(msg)
            return urls
        raise RuntimeError(msg)


def reconnect(connection):
    """Open a new datafind connection based on an existing connection

    This is required because of https://git.ligo.org/lscsoft/glue/issues/1

    Parameters
    ----------
    connection : :class:`~gwdatafind.http.HTTPConnection` or `FflConnection`
        a connection object (doesn't need to be open)

    Returns
    -------
    newconn : :class:`~gwdatafind.http.HTTPConnection` or `FflConnection`
        the new open connection to the same `host:port` server
    """
    if isinstance(connection, FflConnection):
        return type(connection)(connection.ffldir)
    kw = {'context': connection._context} if connection.port != 80 else {}
    return connection.__class__(connection.host, port=connection.port, **kw)


def _type_priority(ifo, ftype, trend=None):
    """Prioritise the given GWF type based on its name or trend status.

    This is essentially an ad-hoc ordering function based on internal knowledge
    of how LIGO does GWF type naming.
    """
    # if looking for a trend channel, prioritise the matching type
    for trendname, trend_regex in [
            ('m-trend', MINUTE_TREND_TYPE),
            ('s-trend', SECOND_TREND_TYPE),
    ]:
        if trend == trendname and trend_regex.match(ftype):
            return 0, len(ftype)

    # otherwise rank this type according to priority
    for reg, prio in {
            HIGH_PRIORITY_TYPE: 1,
            re.compile(r'[A-Z]\d_C'): 6,
            LOW_PRIORITY_TYPE: 10,
            MINUTE_TREND_TYPE: 10,
            SECOND_TREND_TYPE: 10,
    }.items():
        if reg.search(ftype):
            return prio, len(ftype)

    return 5, len(ftype)


def on_tape(*files):
    """Determine whether any of the given files are on tape

    Parameters
    ----------
    *files : `str`
        one or more paths to GWF files

    Returns
    -------
    True/False : `bool`
        `True` if any of the files are determined to be on tape,
        otherwise `False`
    """
    for path in files:
        try:
            if os.stat(path).st_blocks == 0:
                return True
        except AttributeError:  # windows doesn't have st_blocks
            return False
    return False


def _choose_connection(**datafind_kw):
    if os.getenv('LIGO_DATAFIND_SERVER') or datafind_kw.get('host'):
        from gwdatafind import connect
        return connect(**datafind_kw)
    if os.getenv('VIRGODATA'):
        return FflConnection()
    raise RuntimeError("unknown datafind configuration, cannot discover data")


def with_connection(func):
    """Decorate a function to open a new datafind connection if required

    This method will inspect the ``connection`` keyword, and if `None`
    (or missing), will use the ``host`` and ``port`` keywords to open
    a new connection and pass it as ``connection=<new>`` to ``func``.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        if kwargs.get('connection') is None:
            kwargs['connection'] = _choose_connection(host=kwargs.get('host'),
                                                      port=kwargs.get('port'))
        try:
            return func(*args, **kwargs)
        except HTTPException:
            kwargs['connection'] = reconnect(kwargs['connection'])
            return func(*args, **kwargs)
    return wrapped


# -- user methods -------------------------------------------------------------

@with_connection
def find_frametype(channel, gpstime=None, frametype_match=None,
                   host=None, port=None, return_all=False, allow_tape=False,
                   connection=None, on_gaps='error'):
    """Find the frametype(s) that hold data for a given channel

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`
        the channel to be found

    gpstime : `int`, optional
        target GPS time at which to find correct type

    frametype_match : `str`, optional
        regular expression to use for frametype `str` matching

    host : `str`, optional
        name of datafind host to use

    port : `int`, optional
        port on datafind host to use

    return_all : `bool`, optional, default: `False`
        return all found types, default is to return to 'best' match

    allow_tape : `bool`, optional, default: `False`
        do not test types whose frame files are stored on tape (not on
        spinning disk)

    on_gaps : `str`, optional
        action to take when gaps are discovered in datafind coverage,
        default: ``'error'``, i.e. don't match frametypes with gaps.
        Select ``'ignore'`` to ignore gaps, or ``'warn'`` to display
        warnings when gaps are found in a datafind `find_urls` query

    Returns
    -------
    If a single name is given, and `return_all=False` (default):

    frametype : `str`
        name of best match frame type

    If a single name is given, and `return_all=True`:

    types : `list` of `str`
        the list of all matching frame types

    If multiple names are given, the above return types are wrapped into a
    `dict` of `(channel, type_or_list)` pairs.

    Examples
    --------
    >>> from gwpy.io import datafind as io_datafind
    >>> io_datafind.find_frametype('H1:IMC-PWR_IN_OUTPUT', gpstime=1126259462)
    'H1_R'
    >>> io_datafind.find_frametype('H1:IMC-PWR_IN_OUTPUT', gpstime=1126259462,
    ...                            return_all=True)
    ['H1_R', 'H1_C']
    >>> io_datafind.find_frametype(
    ...     ('H1:IMC-PWR_IN_OUTPUT', 'H1:OMC-DCPD_SUM_OUTPUT',
    ...      'H1:GDS-CALIB_STRAIN'),
    ...     gpstime=1126259462, return_all=True))"
    {'H1:GDS-CALIB_STRAIN': ['H1_HOFT_C00'],
     'H1:OMC-DCPD_SUM_OUTPUT': ['H1_R', 'H1_C'],
     'H1:IMC-PWR_IN_OUTPUT': ['H1_R', 'H1_C']}
    """
    # this function is now horrendously complicated to support a large
    # number of different use cases, hopefully the comments are sufficient

    from ..detector import Channel

    # format channel names as list
    if isinstance(channel, (list, tuple)):
        channels = channel
    else:
        channels = [channel]

    # create set() of GWF channel names, and dict map back to user names
    #    this allows users to use nds-style names in this query, e.g.
    #    'X1:TEST.mean,m-trend', and still get results
    chans = {Channel(c).name: c for c in channels}
    names = set(chans.keys())

    # format GPS time(s)
    if isinstance(gpstime, (list, tuple)):
        gpssegment = LigoSegment(*gpstime)
        gpstime = gpssegment[0]
    else:
        gpssegment = None
    if gpstime is not None:
        gpstime = int(to_gps(gpstime))

    # if use gaps post-S5 GPStime, forcibly skip _GRBYYMMDD frametypes at CIT
    if frametype_match is None and gpstime is not None and gpstime > 875232014:
        frametype_match = GRB_TYPE

    # -- go

    match = {}
    ifos = set()  # record IFOs we have queried to prevent duplication

    # loop over set of names, which should get smaller as names are searched
    while names:
        # parse first channel name (to get IFO and channel type)
        try:
            name = next(iter(names))
        except KeyError:
            break
        else:
            chan = Channel(chans[name])

        # parse IFO ID
        try:
            ifo = chan.ifo[0]
        except TypeError:  # chan.ifo is None
            raise ValueError("Cannot parse interferometer prefix from channel "
                             "name %r, cannot proceed with find()" % str(chan))

        # if we've already gone through the types for this IFO, skip
        if ifo in ifos:
            names.pop()
            continue
        ifos.add(ifo)

        types = find_types(ifo, match=frametype_match, trend=chan.type,
                           connection=connection)

        # loop over types testing each in turn
        for ftype in types:
            # find instance of this frametype
            try:
                path = find_latest(ifo, ftype, gpstime=gpstime,
                                   allow_tape=allow_tape,
                                   connection=connection, on_missing='ignore')
            except (RuntimeError, IOError, IndexError):  # something went wrong
                continue

            # check for gaps in the record for this type
            if gpssegment is None:
                gaps = 0
            else:
                cache = find_urls(ifo, ftype, *gpssegment, on_gaps=on_gaps,
                                  connection=connection)
                csegs = cache_segments(cache)
                gaps = abs(gpssegment) - abs(csegs)

            # search the TOC for one frame file and match any channels
            i = 0
            nchan = len(names)
            for n in iter_channel_names(path):
                if n in names:
                    i += 1
                    c = chans[n]  # map back to user-given channel name
                    try:
                        match[c].append((ftype, path, -gaps))
                    except KeyError:
                        match[c] = [(ftype, path, -gaps)]
                    if not return_all:  # match once only
                        names.remove(n)
                    if not names or n == nchan:  # break out of TOC loop
                        break

            if not names:  # break out of ftype loop if all names matched
                break

        try:
            names.pop()
        except KeyError:  # done
            break

    # raise exception if nothing found
    missing = set(channels) - set(match.keys())
    if missing:
        msg = "Cannot locate the following channel(s) in any known frametype"
        if gpstime:
            msg += " at GPS=%d" % gpstime
        msg += ":\n    {}".format('\n    '.join(missing))
        if not allow_tape:
            msg += ("\n[files on tape have not been checked, use "
                    "allow_tape=True for a complete search]")
        raise ValueError(msg)

    # if matching all types, rank based on coverage, tape, and TOC size
    if return_all:
        paths = set(p[1] for key in match for p in match[key])
        rank = {path: (on_tape(path), num_channels(path)) for path in paths}
        # deprioritise types on tape and those with lots of channels
        for key in match:
            match[key].sort(key=lambda x: (x[2],) + rank[x[1]])
        # remove instance paths (just leave channel and list of frametypes)
        ftypes = {key: list(list(zip(*match[key]))[0]) for key in match}
    else:
        ftypes = {key: match[key][0][0] for key in match}

    # if given a list of channels, return a dict
    if isinstance(channel, (list, tuple)):
        return ftypes

    # otherwise just return a list for this type
    return ftypes[str(channel)]


@with_connection
def find_best_frametype(channel, start, end,
                        frametype_match=None, allow_tape=True,
                        connection=None, host=None, port=None):
    """Intelligently select the best frametype from which to read this channel

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`
        the channel to be found

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of period of interest,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of period of interest,
        any input parseable by `~gwpy.time.to_gps` is fine

    host : `str`, optional
        name of datafind host to use

    port : `int`, optional
        port on datafind host to use

    frametype_match : `str`, optiona
        regular expression to use for frametype `str` matching

    allow_tape : `bool`, optional
        do not test types whose frame files are stored on tape (not on
        spinning disk)

    Returns
    -------
    frametype : `str`
        the best matching frametype for the ``channel`` in the
        ``[start, end)`` interval

    Raises
    ------
    ValueError
        if no valid frametypes are found

    Examples
    --------
    >>> from gwpy.io.datafind import find_best_frametype
    >>> find_best_frametype('L1:GDS-CALIB_STRAIN', 1126259460, 1126259464)
    'L1_HOFT_C00'
    """
    try:
        return find_frametype(channel, gpstime=(start, end),
                              frametype_match=frametype_match,
                              allow_tape=allow_tape, on_gaps='error',
                              connection=connection, host=host, port=port)
    except RuntimeError:  # gaps (or something else went wrong)
        ftout = find_frametype(channel, gpstime=(start, end),
                               frametype_match=frametype_match,
                               return_all=True, allow_tape=allow_tape,
                               on_gaps='ignore', connection=connection,
                               host=host, port=port)
        try:
            if isinstance(ftout, dict):
                return {key: ftout[key][0] for key in ftout}
            return ftout[0]
        except IndexError:
            raise ValueError("Cannot find any valid frametypes for channel(s)")


@with_connection
def find_types(observatory, match=None, trend=None,
               connection=None, **connection_kw):
    """Find the available data types for a given observatory.

    See also
    --------
    gwdatafind.http.HTTPConnection.find_types
    FflConnection.find_types
        for details on the underlying method(s)
    """
    return sorted(connection.find_types(observatory, match=match),
                  key=lambda x: _type_priority(observatory, x, trend=trend))


@with_connection
def find_urls(observatory, frametype, start, end, on_gaps='error',
              connection=None, **connection_kw):
    """Find the URLs of files of a given data type in a GPS interval.

    See also
    --------
    gwdatafind.http.HTTPConnection.find_urls
    FflConnection.find_urls
        for details on the underlying method(s)
    """
    return connection.find_urls(observatory, frametype, start, end,
                                on_gaps=on_gaps)


@with_connection
def find_latest(observatory, frametype, gpstime=None, allow_tape=False,
                connection=None, **connection_kw):
    """Find the path of the latest file of a given data type.

    See also
    --------
    gwdatafind.http.HTTPConnection.find_latest
    FflConnection.find_latest
        for details on the underlying method(s)
    """
    observatory = observatory[0]
    try:
        if gpstime is not None:
            gpstime = int(to_gps(gpstime))
            path = find_urls(observatory, frametype, gpstime, gpstime+1,
                             on_gaps='ignore', connection=connection)[-1]
        else:
            path = connection.find_latest(observatory, frametype,
                                          on_missing='ignore')[-1]
    except (IndexError, RuntimeError):
        raise RuntimeError(
            "no files found for {}-{}".format(observatory, frametype))

    path = urlparse(path).path
    if not allow_tape and on_tape(path):
        raise IOError("Latest frame file for {}-{} is on tape "
                      "(pass allow_tape=True to force): "
                      "{}".format(observatory, frametype, path))
    return path
