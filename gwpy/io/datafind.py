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

from glue.lal import CacheEntry

from ..time import to_gps
from ..utils import (shell, with_import)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

S6_HOFT_NAME = re.compile('\A(H|L)1:LDAS-STRAIN\Z')
S6_RECOLORED_TYPE = re.compile('\AT1200307_V4_EARLY_RECOLORED_V2\Z')
SECOND_TREND_TYPE = re.compile('\A(.*_)?T\Z')  # T or anything ending in _T
MINUTE_TREND_TYPE = re.compile('\A(.*_)?M\Z')  # M or anything ending in _M


@with_import('glue.datafind')
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
    port = port and int(port)
    if port is not None and port != 80:
        cert, key = datafind.find_credential()
        return datafind.GWDataFindHTTPSConnection(
            host=host, port=port, cert_file=cert, key_file=key)
    else:
        return datafind.GWDataFindHTTPConnection(host=host, port=port)


def find_frametype(channel, gpstime=None, frametype_match=None,
                   host=None, port=None, return_all=False, exclude_tape=False):
    """Find the frametype(s) that hold data for a given channel

    Parameters
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
    exclude_tape : `bool`, optional, default: `False`
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
    if gpstime is not None:
        gpstime = to_gps(gpstime).seconds
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
                    not exclude_tape or not on_tape(frame)):
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
    if exclude_tape:
        msg = "Cannot locate %r in any known frametype that isn't on tape"
    else:
        msg = "Cannot locate %r in any known frametype"
    if gpstime:
        msg += " at GPS=%d" % gpstime
    if len(found) == 0:
        raise ValueError(msg % name)
    else:
        return found


@with_import('lalframe')
def num_channels(framefile):
    """Find the total number of channels in this framefile

    Parameters
    ----------
    framefile : `str`
        path to GWF-format file on disk

    Returns
    -------
    n : `int`
        the total number of channels found in the table of contents for this
        file

    Notes
    -----
    This method requires LALFrame
    """
    frfile = lalframe.FrameUFrFileOpen(framefile, "r")
    frtoc = lalframe.FrameUFrTOCRead(frfile)
    return sum(
        getattr(lalframe, 'FrameUFrTOCQuery%sN' % type_.title())(frtoc) for
        type_ in ['adc', 'proc', 'sim'])


@with_import('lalframe')
def get_channel_type(channel, framefile):
    """Find the channel type in a given frame file

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`
        name of data channel to find
    framefile : `str`
        path of GWF file in which to search

    Returns
    -------
    ctype : `str`
        the type of the channel ('adc', 'sim', or 'proc') if the
        channel exists in the table-of-contents for the given frame,
        otherwise `False`
    """
    name = str(channel)
    # read frame and table of contents
    frfile = lalframe.FrameUFrFileOpen(framefile, "r")
    frtoc = lalframe.FrameUFrTOCRead(frfile)
    for type_ in ['sim', 'proc', 'adc']:
        query = getattr(lalframe, 'FrameUFrTOCQuery%sName' % type_.title())
        i = 0
        while True:
            try:
                c = query(frtoc, i)
            except RuntimeError:
                break
            else:
                if c == name:
                    return type_
            i += 1
    return False


def channel_in_frame(channel, framefile):
    """Determine whether a channel is stored in this framefile

    Parameters
    ----------
    channel : `str`
        name of channel to find
    framefile : `str`
        path of GWF file to test

    Returns
    -------
    inframe : `bool`
        whether this channel is included in the table of contents for
        the given framefile
    """
    name = str(channel)
    try:
        out = shell.call(['FrChannels', framefile])[0]
    except (OSError, shell.CalledProcessError):
        return get_channel_type(channel, framefile) is not False
    else:
        for line in out.splitlines():
            if line.split(' ')[0] == name:
                return True
        return False


def find_best_frametype(channel, start, end, urltype='file',
                        host=None, port=None, allow_tape=True):
    """Intelligently select the best frametype from which to read this channel
    """
    start = to_gps(start).seconds
    end = to_gps(end).seconds
    frametype = find_frametype(channel, gpstime=start, host=host, port=port,
                               exclude_tape=not allow_tape)
    connection = connect(host=host, port=port)
    try:
        cache = connection.find_frame_urls(channel[0], frametype,
                                           start, end, urltype=urltype,
                                           on_gaps='error')
        if not allow_tape and on_tape(*cache):
            raise RuntimeError()
    except RuntimeError:
        alltypes = find_frametype(channel, gpstime=start, host=host, port=port,
                                  return_all=True, exclude_tape=not allow_tape)
        cache = [(ft, connection.find_frame_urls(
            channel[0], ft, start, end, urltype=urltype,
            on_gaps='ignore')) for ft in alltypes]
        if not allow_tape:
            cache = [ftc for ftc in cache if not on_tape(*ftc[1])]
        cache.sort(key=lambda x:
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
    *files : `str`, `~glue.lal.CacheEntry`
        one or more paths to GWF files

    Returns
    -------
    True/False : `bool`
        `True` if any of the files are determined to be on tape,
        otherwise `False`
    """
    for f in files:
        if isinstance(f, CacheEntry):
            f = f.path
        if os.stat(f).st_blocks == 0:
            return True
    return False
