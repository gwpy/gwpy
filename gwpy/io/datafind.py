# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of GWpy.

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

"""User-friendly extensions to :mod:`glue.datafind`

The functions in this module mainly focus on matching a channel name to
one or more frametypes that contain data for that channel.
"""

import os.path
import re

from ..time import to_gps
from .cache import cache_segments
from .gwf import (num_channels, iter_channel_names)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# special-case frame types
SECOND_TREND_TYPE = re.compile(r'\A(.*_)?T\Z')  # T or anything ending in _T
MINUTE_TREND_TYPE = re.compile(r'\A(.*_)?M\Z')  # M or anything ending in _M
GRB_TYPE = re.compile(r'^(?!.*_GRB\d{6}([A-Z])?$)')
HIGH_PRIORITY_TYPE = re.compile(
    r'[A-Z]\d_HOFT_C\d\d(_T\d{7}_v\d)?\Z'  # X1_HOFT_CXY
)
LOW_PRIORITY_TYPE = re.compile(
    r'(_GRB\d{6}([A-Z])?\Z|'  # endswith _GRBYYMMDD{A}
    r'\AT1200307_V4_EARLY_RECOLORED_V2\Z)'  # annoying recoloured HOFT type
)


def connect(host=None, port=None):
    """Open a new datafind connection

    Parameters
    ----------
    host : `str`
        name of datafind server to query

    port : `int`
        port of datafind server on host

    Returns
    -------
    connection : :class:`~glue.datafind.GWDataFindHTTPConnection`
        the new open connection
    """
    from glue import datafind

    port = port and int(port)
    if port is not None and port != 80:
        cert, key = datafind.find_credential()
        return datafind.GWDataFindHTTPSConnection(
            host=host, port=port, cert_file=cert, key_file=key)

    return datafind.GWDataFindHTTPConnection(host=host, port=port)


def reconnect(connection):
    """Open a new datafind connection based on an existing connection

    This is required because of https://git.ligo.org/lscsoft/glue/issues/1

    Parameters
    ----------
    connection : :class:`~glue.datafind.GWDataFindHTTPConnection`
        a connection object (doesn't need to be open)

    Returns
    -------
    newconn : :class:`~glue.datafind.GWDataFindHTTPConnection`
        the new open connection to the same `host:port` server
    """
    return connect(connection.host, connection.port)


def find_frametype(channel, gpstime=None, frametype_match=None,
                   host=None, port=None, return_all=False, allow_tape=False,
                   urltype='file', on_gaps='error'):
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

    urltype : `str`, optional
        scheme of URL to return, default is ``'file'``

    on_gaps : `str`, optional
        action to take when gaps are discovered in datafind coverage,
        default: ``'error'``, i.e. don't match frametypes with gaps.
        Select ``'ignore'`` to ignore gaps, or ``'warn'`` to display
        warnings when gaps are found in a datafind `find_frame_urls` query

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
        from ..segments import Segment
        gpssegment = Segment(*gpstime)
        gpstime = gpssegment[0]
    else:
        gpssegment = None
    if gpstime is not None:
        gpstime = to_gps(gpstime).gpsSeconds

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

        # connect and find list of all frame types
        connection = connect(host, port)
        types = connection.find_types(ifo, match=frametype_match)

        # sort frametypes by likely requirements (to speed up matching)
        def _type_key(ftype):
            # HOFT types typically have small channel lists (so quick search)
            if HIGH_PRIORITY_TYPE.match(ftype):  # HOFT types are small
                prio = 1
            # these types are bogus, or just unhelpful
            elif LOW_PRIORITY_TYPE.match(ftype):
                prio = 10

            # if channel is trend, promote trend type (otherwise demote)
            elif chan.type == 'm-trend' and MINUTE_TREND_TYPE.match(ftype):
                prio = 0
            elif MINUTE_TREND_TYPE.match(ftype):
                prio = 10
            elif chan.type == 's-trend' and SECOND_TREND_TYPE.match(ftype):
                prio = 0
            elif SECOND_TREND_TYPE.match(ftype):
                prio = 10

            # demote commissioning frames for LIGO
            elif ftype == '{}_C'.format(chan.ifo):
                prio = 6

            # otherwise give a middle score
            else:
                prio = 5

            # use score and length of name, shorter names are typically better
            return (prio, len(ftype))

        types.sort(key=_type_key)

        # loop over types testing each in turn
        for ftype in types:
            # find instance of this frametype
            try:
                connection = reconnect(connection)
                path = _find_latest_frame(connection, ifo, ftype,
                                          gpstime=gpstime,
                                          allow_tape=allow_tape)
            except (RuntimeError, IOError):  # something went wrong
                continue

            # check for gaps in the record for this type
            if gpssegment is None:
                gaps = 0
            else:
                connection = reconnect(connection)
                cache = connection.find_frame_urls(ifo, ftype, *gpssegment,
                                                   urltype=urltype,
                                                   on_gaps=on_gaps)
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


def find_best_frametype(channel, start, end, urltype='file',
                        host=None, port=None, frametype_match=None,
                        allow_tape=True):
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

    urltype : `str`, optional
        scheme of URL to return, default is ``'file'``

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
        return find_frametype(channel, gpstime=(start, end), host=host,
                              port=port, frametype_match=frametype_match,
                              allow_tape=allow_tape, urltype=urltype,
                              on_gaps='error')
    except RuntimeError:  # gaps (or something else went wrong)
        ftout = find_frametype(channel, gpstime=(start, end), host=host,
                               port=port, frametype_match=frametype_match,
                               return_all=True, allow_tape=allow_tape,
                               urltype=urltype, on_gaps='ignore')
        try:
            if isinstance(ftout, dict):
                return {key: ftout[key][0] for key in ftout}
            return ftout[0]
        except IndexError:
            raise ValueError("Cannot find any valid frametypes for channel(s)")


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
        if os.stat(path).st_blocks == 0:
            return True
    return False


# -- utilities ----------------------------------------------------------------

def _find_latest_frame(connection, ifo, frametype, gpstime=None,
                       allow_tape=False):
    """Find the latest framepath for a given frametype
    """
    ifo = ifo[0]
    if gpstime is not None:
        gpstime = int(to_gps(gpstime))
    try:
        if gpstime is None:
            frame = connection.find_latest(ifo, frametype, urltype='file')[0]
        else:
            frame = connection.find_frame_urls(ifo, frametype, gpstime,
                                               gpstime, urltype='file',
                                               on_gaps='ignore')[0]
    except (IndexError, RuntimeError):
        raise RuntimeError("No frames found for {}-{}".format(ifo, frametype))
    else:
        if not os.access(frame.path, os.R_OK):
            raise IOError("Latest frame file for {}-{} is unreadable: "
                          "{}".format(ifo, frametype, frame.path))
        if not allow_tape and on_tape(frame.path):
            raise IOError("Latest frame file for {}-{} is on tape "
                          "(pass allow_tape=True to force): "
                          "{}".format(ifo, frametype, frame.path))
        return frame.path
