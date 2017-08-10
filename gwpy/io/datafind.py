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

"""User-friendly extensions to `glue.datafind`
"""

import os.path
import re

from ..time import to_gps
from .gwf import (num_channels, channel_in_frame)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

S6_HOFT_NAME = re.compile('\A(H|L)1:LDAS-STRAIN\Z')
S6_RECOLORED_TYPE = re.compile('\AT1200307_V4_EARLY_RECOLORED_V2\Z')
SECOND_TREND_TYPE = re.compile('\A(.*_)?T\Z')  # T or anything ending in _T
MINUTE_TREND_TYPE = re.compile('\A(.*_)?M\Z')  # M or anything ending in _M


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
    connection : `~glue.datafind.GWDataFindHTTPConnection`
        the new open connection
    """
    from glue import datafind

    port = port and int(port)
    if port is not None and port != 80:
        cert, key = datafind.find_credential()
        return datafind.GWDataFindHTTPSConnection(
            host=host, port=port, cert_file=cert, key_file=key)
    else:
        return datafind.GWDataFindHTTPConnection(host=host, port=port)


def find_frametype(channel, gpstime=None, frametype_match=None,
                   host=None, port=None, return_all=False, allow_tape=False):
    """Find the frametype(s) that hold data for a given channel

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`
        the channel to be found

    gpstime : `int`, optional
        target GPS time at which to find correct type

    frametype_match : `str`, optiona
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

    Returns
    -------
    frametype : `str`
        if `return_all` is `False`, name of best match frame type
    types : `list` of `str`
        if `return_all` is `True`, the list of all matching frame types
    """
    from ..detector import Channel
    channel = Channel(channel)
    name = channel.name
    if not channel.ifo:
        raise ValueError("Cannot parse interferometer prefix from channel "
                         "name %r, cannot proceed with find()" % name)
    if gpstime is not None:
        gpstime = to_gps(gpstime).gpsSeconds
    connection = connect(host, port)
    types = connection.find_types(channel.ifo[0], match=frametype_match)
    # get reference frame for all types
    frames = []
    for ft in types:
        try:
            if gpstime is None:
                frame = connection.find_latest(
                    channel.ifo[0], ft, urltype='file')[0]
            else:
                frame = connection.find_frame_urls(
                    channel.ifo[0], ft, gpstime, gpstime, urltype='file',
                    on_gaps='ignore')[0]
        except (IndexError, RuntimeError):
            continue
        else:
            if os.access(frame.path, os.R_OK) and (
                    allow_tape or not on_tape(frame.path)):
                frames.append((ft, frame.path))
    # sort frames by allocated block size and regular size
    # (to put frames on tape at the bottom of the list)
    frames.sort(key=lambda x: (on_tape(x[1]), num_channels(x[1])))
    # if looking for LDAS-STRAIN, put recoloured types at the end
    if S6_HOFT_NAME.match(name):
        frames.sort(key=lambda x: S6_RECOLORED_TYPE.match(x[0]) and 2 or 1)

    # need to handle trends as a special case
    if channel.type == 'm-trend':
        frames.sort(key=lambda x: MINUTE_TREND_TYPE.match(x[0]) and 1 or 2)
        # if no minute-trend types found, force an error
        if frames and not MINUTE_TREND_TYPE.match(frames[0][0]):
            frames = []
    elif channel.type == 's-trend':
        frames.sort(key=lambda x: SECOND_TREND_TYPE.match(x[0]) and 1 or 2)
        # if no second-trend types found, force an error
        if frames and not SECOND_TREND_TYPE.match(frames[0][0]):
            frames = []

    # search each frametype for the given channel
    found = []
    for ft, path in frames:
        inframe = channel_in_frame(name, path)
        if inframe and not return_all:
            return ft
        elif inframe:
            found.append(ft)
    msg = "Cannot locate %r in any known frametype" % name
    if gpstime:
        msg += " at GPS=%d" % gpstime
    if not allow_tape:
        msg += (" [those files on tape have not been checked, use "
                "allow_tape=True to perform a complete search]")
    if len(found) == 0:
        raise ValueError(msg)
    else:
        return found


def find_best_frametype(channel, start, end, urltype='file',
                        host=None, port=None, allow_tape=True):
    """Intelligently select the best frametype from which to read this channel
    """
    start = to_gps(start).gpsSeconds
    end = to_gps(end).gpsSeconds
    frametype = find_frametype(channel, gpstime=start, host=host, port=port,
                               allow_tape=allow_tape)
    connection = connect(host=host, port=port)
    try:
        cache = connection.find_frame_urls(channel[0], frametype,
                                           start, end, urltype=urltype,
                                           on_gaps='error')
        if not allow_tape and on_tape(*cache.pfnlist()):
            raise RuntimeError()
    except RuntimeError:
        alltypes = find_frametype(channel, gpstime=start, host=host, port=port,
                                  return_all=True, allow_tape=allow_tape)
        cache = [(ft, connection.find_frame_urls(
            channel[0], ft, start, end, urltype=urltype,
            on_gaps='ignore')) for ft in alltypes]
        if not allow_tape:
            cache = [ftc for ftc in cache if not on_tape(*ftc[1].pfnlist())]
        cache.sort(
            key=lambda x:
            len(x[1]) and -abs(x[1].to_segmentlistdict().values()[0]) or 0)
        try:
            return cache[0][0]
        except IndexError:
            raise ValueError("Cannot find any valid frametypes for %r"
                             % channel)
    else:
        return frametype


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
    for f in files:
        if os.stat(f).st_blocks == 0:
            return True
    return False
