# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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
package (a dependency of ``gwpy``), and either the ``GW_DATAFIND_SERVER``
(or legacy ``LIGO_DATAFIND_SERVER``) environment variable to be set,
or the ``host`` keyword must be passed to :func:`find_urls` and friends.

Data discovery using the Virgo FFL system requires the ``FFLPATH`` environment
variable to point to the directory containing FFL files, **or** the
``VIRGODATA`` environment variable to point to a directory containing an
``ffl` subdirectory, which contains FFL files.
"""

import os
import os.path
import re
import warnings
from collections import defaultdict
from functools import wraps
from unittest import mock

import gwdatafind

from ligo.segments import segment as LigoSegment

from ..time import to_gps
from . import ffldatafind
from .cache import cache_segments
from .gwf import (num_channels, iter_channel_names)
from .utils import file_path

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

SINGLE_IFO_OBSERVATORY = re.compile("^[A-Z][0-9]$")

# special-case frame types
LIGO_SECOND_TREND_TYPE = re.compile(r'\A(.*_)?T\Z')  # T or *_T
LIGO_MINUTE_TREND_TYPE = re.compile(r'\A(.*_)?M\Z')  # M or *_M
VIRGO_SECOND_TREND_TYPE = re.compile(r"\A(.*_)?[Tt]rend\Z")  # trend or *_trend
GRB_TYPE = re.compile(r'^(?!.*_GRB\d{6}([A-Z])?$)')  # *_GRBYYMMDD{A}
HIGH_PRIORITY_TYPE = re.compile("({})".format("|".join((
    r'\A[A-Z]\d_HOFT_C\d\d(_T\d{7}_v\d)?\Z',  # X1_HOFT_CXY
    r'\AV1Online\Z',
    r'\AV1O[0-9]+([A-Z]+)?Repro[0-9]+[A-Z]+\Z',  # V1OXReproXY
))))
LOW_PRIORITY_TYPE = re.compile("({})".format("|".join((
    r'_GRB\d{6}([A-Z])?\Z',  # endswith _GRBYYMMDD{A}
    r'_bck\Z',  # endswith _bck
    r'\AT1200307_V4_EARLY_RECOLORED_V2\Z',  # annoying recoloured HOFT type
))))


# -- utilities ----------------------------------------------------------------

def _type_priority(ifo, ftype, trend=None):
    """Prioritise the given GWF type based on its name or trend status.

    This is essentially an ad-hoc ordering function based on internal knowledge
    of how LIGO does GWF type naming.
    """
    # if looking for a trend channel, prioritise the matching type
    for trendname, trend_regex in [
            ('m-trend', LIGO_MINUTE_TREND_TYPE),
            ('s-trend', LIGO_SECOND_TREND_TYPE),
            ('s-trend', VIRGO_SECOND_TREND_TYPE),
    ]:
        if trend == trendname and trend_regex.match(ftype):
            return 0, len(ftype)

    # otherwise rank this type according to priority
    for reg, prio in {
            HIGH_PRIORITY_TYPE: 1,
            re.compile(r'[A-Z]\d_C'): 6,
            LOW_PRIORITY_TYPE: 10,
            LIGO_MINUTE_TREND_TYPE: 10,
            LIGO_SECOND_TREND_TYPE: 10,
            VIRGO_SECOND_TREND_TYPE: 10,
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
        stat = os.stat(path)
        try:
            if stat.st_blocks == 0:
                return True
        except AttributeError:  # windows doesn't have st_blocks
            return False
    return False


def _gwdatafind_module(**datafind_kw):
    """Return the appropriate GWDataFind-like API based on the environment

    This allows switching to the hacky `gwpy.io.ffldatafind` replacement
    module to enable a GWDataFind-like interface for direct FFL data
    discovery at Virgo.
    """
    # GWDataFind
    if (
        os.getenv('GWDATAFIND_SERVER')
        or os.getenv('LIGO_DATAFIND_SERVER')
        or datafind_kw.get('host')
    ):
        return gwdatafind

    # FFL
    try:
        ffldatafind._get_ffl_basedir()
    except KeyError:  # failed to discover FFL directories
        raise RuntimeError(
            "unknown datafind configuration, cannot discover data",
        )
    return ffldatafind


def _select_gwdatafind_mod(func):
    """Decorate a function to see the right ``gwdatafind`` API.

    This exists only to allow on-the-fly replacing of the actual `gwdatafind`
    with :mod:`gwpy.io.ffldatafind` if it looks like we are trying to find
    data from FFL files.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        # replace the 'gwdatafind' module in the function namespace
        # with the API we need for this call
        with mock.patch.dict(func.__globals__):
            func.__globals__["gwdatafind"] = _gwdatafind_module(**kwargs)
            return func(*args, **kwargs)

    return wrapped


def _parse_ifos_and_trends(chans):
    """Parse ``(ifo, trend)`` pairs from this list of channels
    """
    from ..detector import Channel
    found = set()
    for name in chans:
        chan = Channel(name)
        try:
            found.add((chan.ifo[0], chan.type))
        except TypeError:  # chan.ifo is None
            raise ValueError(
                "Cannot parse interferometer prefix from channel name "
                f"'{chan}', cannot proceed with find()",
            )
    return found


def _find_gaps(ifo, frametype, segment, on_gaps):
    """Discover gaps in a datafind/ffl archive for the given ifo/type

    Returns
    -------
    gaps : `int`
        the cumulative size of all gaps in the relevant archive
    """
    if segment is None:
        return 0
    cache = find_urls(
        ifo,
        frametype,
        *segment,
        on_gaps=on_gaps,
    )
    csegs = cache_segments(cache)
    return max(0, abs(segment) - abs(csegs))


def _error_missing_channels(required, found, gpstime, allow_tape):
    """Raise an exception if required channels are not found
    """
    missing = set(required) - set(found)

    if not missing:  # success
        return

    # failure
    msg = "Cannot locate the following channel(s) in any known frametype"
    if gpstime:
        msg += f" at GPS={gpstime}"
    msg += ":\n    " + "\n    ".join(missing)
    if not allow_tape:
        msg += ("\n[files on tape have not been checked, use "
                "allow_tape=True for a complete search]")
    raise ValueError(msg)


def _rank_types(match):
    """Rank and sort the matched frametypes according to some criteria

    ``matches`` is a dict of (channel, [(type, gwf, gapsize), ...])
    entries.
    """
    paths = set(typetuple[1] for key in match for typetuple in match[key])
    rank = {path: (on_tape(path), num_channels(path)) for path in paths}
    # deprioritise types on tape and those with lots of channels
    for key in match:
        match[key].sort(key=lambda x: (-x[2],) + rank[x[1]])


# -- user methods -------------------------------------------------------------

@_select_gwdatafind_mod
def find_frametype(channel, gpstime=None, frametype_match=None,
                   host=None, port=None, return_all=False, allow_tape=False,
                   on_gaps='error'):
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
    channels = {c: Channel(c).name for c in channels}
    names = {val: key for key, val in channels.items()}

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

    match = defaultdict(list)
    searched = set()

    for ifo, trend in _parse_ifos_and_trends(channels):
        # find all types (prioritising trends if we need to)
        types = find_types(
            ifo,
            match=frametype_match,
            trend=trend,
        )

        # loop over types testing each in turn
        for ftype in types:

            # if we've already search this type for this IFO,
            # don't do it again
            if (ifo, ftype) in searched:
                continue

            # find instance of this frametype
            try:
                path = find_latest(
                    ifo,
                    ftype,
                    gpstime=gpstime,
                    allow_tape=allow_tape,
                    on_missing='ignore',
                )
            except (RuntimeError, IOError, IndexError):  # something went wrong
                continue

            # check for gaps in the record for this type
            gaps = _find_gaps(ifo, ftype, gpssegment, on_gaps)

            # search the TOC for one frame file and match any channels
            found = 0
            nchan = len(names)
            try:
                for n in iter_channel_names(path):
                    if n in names:  # frametype includes this channel!
                        # count how many channels we have found in this type
                        found += 1

                        # record the match using the user-given channel name
                        match[names[n]].append((ftype, path, gaps))

                        # if only matching once, don't search other types
                        # for this channel
                        if not return_all:
                            names.pop(n)

                        if found == nchan:  # all channels have been found
                            break
            except RuntimeError as exc:  # failed to open file (probably)
                warnings.warn(
                    f"failed to read channels for type {ftype!r}: {exc}:",
                )
                continue

            # record this type as having been searched
            searched.add((ifo, ftype))

            if not names:  # if all channels matched, stop
                break

    # raise exception if one or more channels were not found
    _error_missing_channels(channels, match.keys(), gpstime, allow_tape)

    # rank types (and pick best if required)
    _rank_types(match)

    # and format as a dict for each channel
    output = {key: list(list(zip(*match[key]))[0]) for key in match}
    if not return_all:  # reduce the list-of-one to a single element
        output = {key: val[0] for key, val in output.items()}

    # if given a list of channels, return the dict
    if isinstance(channel, (list, tuple)):
        return output

    # otherwise just return the result for the given channel
    return output[str(channel)]


@_select_gwdatafind_mod
def find_best_frametype(channel, start, end,
                        frametype_match=None, allow_tape=True,
                        host=None, port=None):
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
                              host=host, port=port)
    except RuntimeError:  # gaps (or something else went wrong)
        ftout = find_frametype(channel, gpstime=(start, end),
                               frametype_match=frametype_match,
                               return_all=True, allow_tape=allow_tape,
                               on_gaps='ignore', host=host, port=port)
        try:
            if isinstance(ftout, dict):
                return {key: ftout[key][0] for key in ftout}
            return ftout[0]
        except IndexError:
            raise ValueError("Cannot find any valid frametypes for channel(s)")


@_select_gwdatafind_mod
def find_types(observatory, match=None, trend=None, **kwargs):
    """Find the available data types for a given observatory.

    See also
    --------
    gwdatafind.find_types
    """
    return sorted(
        gwdatafind.find_types(observatory, match=match, **kwargs),
        key=lambda x: _type_priority(observatory, x, trend=trend),
    )


@_select_gwdatafind_mod
def find_urls(observatory, frametype, start, end, on_gaps='error', **kwargs):
    """Find the URLs of files of a given data type in a GPS interval.

    See also
    --------
    gwdatafind.find_urls
    """
    return gwdatafind.find_urls(
        observatory,
        frametype,
        start,
        end,
        on_gaps=on_gaps,
        **kwargs,
    )


@_select_gwdatafind_mod
def find_latest(
    observatory,
    frametype,
    gpstime=None,
    allow_tape=False,
    **kwargs,
):
    """Find the path of the latest file of a given data type.

    See also
    --------
    gwdatafind.find_latest
    """
    if SINGLE_IFO_OBSERVATORY.match(observatory):
        observatory = observatory[0]
    try:
        if gpstime is not None:
            gpstime = int(to_gps(gpstime))
            path = find_urls(
                observatory,
                frametype,
                gpstime,
                gpstime+1,
                on_gaps='ignore',
            )[-1]
        else:
            path = gwdatafind.find_latest(
                observatory,
                frametype,
                on_missing='ignore',
            )[-1]
    except (IndexError, RuntimeError):
        raise RuntimeError(f"no files found for {observatory}-{frametype}")

    path = file_path(path)
    if not allow_tape and on_tape(path):
        raise IOError(
            f"Latest frame file for {observatory}-{frametype} is on tape "
            f"(pass allow_tape=True to force): {path}",
        )
    return path
