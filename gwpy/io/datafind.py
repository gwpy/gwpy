# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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
files (either directly, or via NFS/fuse).

Data discovery using the DataFind service requires the `gwdatafind` Python
package (a dependency of ``gwpy``), and either the ``GW_DATAFIND_SERVER``
(or legacy ``LIGO_DATAFIND_SERVER``) environment variable to be set,
or the ``host`` keyword must be passed to :func:`find_urls` and friends.

Data discovery using the Virgo FFL system requires the ``FFLPATH`` environment
variable to point to the directory containing FFL files, **or** the
``VIRGODATA`` environment variable to point to a directory containing an
``ffl`` subdirectory, which contains FFL files.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from contextlib import nullcontext
from functools import (
    partial,
    wraps,
)
from math import ceil
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)
from unittest import mock

import gwdatafind
from igwn_segments import segment as LigoSegment  # noqa: N812
from urllib3.util import parse_url

from ..detector import Channel
from ..testing.errors import NETWORK_ERROR
from ..time import to_gps
from . import ffldatafind
from .cache import cache_segments
from .gwf import (
    iter_channel_names,
    num_channels,
)
from .remote import (
    download_file,
    is_remote,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Mapping,
    )
    from types import ModuleType
    from typing import (
        Literal,
        ParamSpec,
        SupportsFloat,
        TypeVar,
    )

    from ..segments import Segment
    from ..time import SupportsToGps

    P = ParamSpec("P")
    T = TypeVar("T")

    ChannelLike = TypeVar("ChannelLike", bound=str | Channel)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "find_best_frametype",
    "find_frametype",
    "find_latest",
    "find_types",
    "find_urls",
    "on_tape",
]

logger = logging.getLogger(__name__)

SINGLE_IFO_OBSERVATORY = re.compile("^[A-Z][0-9]$")

GPS_END_S5 = 875232014

# special-case frame types
LIGO_SECOND_TREND_TYPE = re.compile(r"\A(.*_)?T\Z")  # T or *_T
LIGO_MINUTE_TREND_TYPE = re.compile(r"\A(.*_)?M\Z")  # M or *_M
VIRGO_SECOND_TREND_TYPE = re.compile(r"\A(.*_)?[Tt]rend\Z")  # trend or *_trend
NOT_GRB_TYPE = re.compile(r"^(?!.*_GRB\d{6}([A-Z])?$)")  # *_GRBYYMMDD{A}
HIGH_PRIORITY_TYPE = re.compile("({})".format("|".join((  # noqa: FLY002
    r"\A[A-Z]\d_GWOSC_O([0-9]+)([a-z])?_[0-9]+KHZ_R[0-9]+",  # X1_GWOSC_OX_NKHZ_RX
    r"\A[A-Z]\d_HOFT_C\d\d(_T\d{7}_v\d)?\Z",  # X1_HOFT_CXY
    r"\AV1Online\Z",
    r"\AHoftOnline\Z",
    r"\AV1O[0-9]+([A-Z]+)?Repro[0-9]+[A-Z]+\Z",  # V1OXReproXY
))))
LOW_PRIORITY_TYPE = re.compile("({})".format("|".join((  # noqa: FLY002
    r"_GRB\d{6}([A-Z])?\Z",  # endswith _GRBYYMMDD{A}
    r"_bck\Z",  # endswith _bck
    r"\AT1200307_V4_EARLY_RECOLORED_V2\Z",  # annoying recoloured HOFT type
))))


# -- utilities -----------------------

def _type_priority(
    ftype: str,
    trend: str | None = None,
) -> tuple[int, int]:
    """Prioritise the given GWF type based on its name or trend status.

    This is essentially an ad-hoc ordering function based on internal knowledge
    of how LIGO does GWF type naming.
    """
    # if looking for a trend channel, prioritise the matching type
    for trendname, trend_regex in [
        ("m-trend", LIGO_MINUTE_TREND_TYPE),
        ("s-trend", LIGO_SECOND_TREND_TYPE),
        ("s-trend", VIRGO_SECOND_TREND_TYPE),
    ]:
        if trend == trendname and trend_regex.match(ftype):
            return 0, len(ftype)

    # otherwise rank this type according to priority
    for reg, prio in {
        HIGH_PRIORITY_TYPE: 1,
        re.compile(r"[A-Z]\d_C"): 6,
        LOW_PRIORITY_TYPE: 10,
        LIGO_MINUTE_TREND_TYPE: 10,
        LIGO_SECOND_TREND_TYPE: 10,
        VIRGO_SECOND_TREND_TYPE: 10,
    }.items():
        if reg.search(ftype):
            return prio, len(ftype)

    return 5, len(ftype)


def on_tape(*files: str) -> bool:
    """Determine whether any of the given files are on tape.

    Parameters
    ----------
    *files : `str`
        one or more paths to GWF files

    Returns
    -------
    ontape : `bool`
        `True` if any of the files are determined to be on tape,
        otherwise `False`.

    Notes
    -----
    A returned value of `False` does not necessarily mean that none of the
    paths are on tape, just that this function couldn't say that they
    definitely *are* on tape.
    """
    for path in files:
        if is_remote(path):  # we can't inspect remote files
            return False
        url = parse_url(path)
        if not url.path:
            return False
        try:
            stat = Path(url.path).stat()
        except FileNotFoundError:
            return False
        try:
            return stat.st_blocks == 0
        except AttributeError:  # windows doesn't have st_blocks
            return False
    return False


def _gwdatafind_module(**datafind_kw) -> ModuleType:
    """Return the appropriate GWDataFind-like API based on the environment.

    This allows switching to the hacky `gwpy.io.ffldatafind` replacement
    module to enable a GWDataFind-like interface for direct FFL data
    discovery at Virgo.
    """
    # GWDataFind
    if (
        os.getenv("GWDATAFIND_SERVER")
        or os.getenv("LIGO_DATAFIND_SERVER")
        or datafind_kw.get("host")
    ):
        return gwdatafind

    # FFL
    try:
        ffldatafind._get_ffl_basedir()  # noqa: SLF001
    except KeyError:  # failed to discover FFL directories
        msg = "unknown datafind configuration, cannot discover data"
        raise RuntimeError(msg) from None
    return ffldatafind


def _select_gwdatafind_mod(func: Callable[P, T]) -> Callable[P, T]:
    """Decorate a function to see the right ``gwdatafind`` API.

    This exists only to allow on-the-fly replacing of the actual `gwdatafind`
    with :mod:`gwpy.io.ffldatafind` if it looks like we are trying to find
    data from FFL files.
    """
    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        # replace the 'gwdatafind' module in the function namespace
        # with the API we need for this call
        with mock.patch.dict(func.__globals__):
            func.__globals__["gwdatafind"] = _gwdatafind_module(**kwargs)
            return func(*args, **kwargs)

    return wrapped


def _parse_ifos_and_trends(
    chans: Iterable[str | Channel],
) -> set[tuple[str, str | None]]:
    """Parse ``(ifo, trend)`` pairs from this list of channels."""
    found = set()
    for name in chans:
        chan = Channel(name)
        if (ifo := chan.ifo) is None:
            msg = (
                "Cannot parse interferometer prefix from channel name "
                f"'{chan}', cannot proceed with find()"
            )
            raise ValueError(msg)
        found.add((ifo[0], chan.type))
    return found


def _find_gaps(
    ifo: str,
    frametype: str,
    segment: Segment | None,
    **kwargs,
) -> float:
    """Discover gaps in a datafind/ffl archive for the given ifo/type.

    Returns
    -------
    gaps : `float`
        The cumulative size of all gaps in the relevant archive.
    """
    if segment is None:
        return 0
    cache = find_urls(
        ifo,
        frametype,
        *segment,
        **kwargs,
    )
    csegs = cache_segments(cache)
    return max(0, abs(segment) - abs(csegs))


def _error_missing_channels(
    required: Iterable[ChannelLike],
    found: Iterable[ChannelLike],
    gpstime: SupportsFloat | None,
    *,
    allow_tape: bool,
) -> None:
    """Raise an exception if required channels are not found."""
    missing = set(map(str, required)) - set(map(str, found))

    if not missing:  # success
        return

    # failure
    msg = "Cannot locate the following channel(s) in any known frametype"
    if gpstime:
        msg += f" at GPS={gpstime}"
    msg += ":\n    " + "\n    ".join(missing)
    if not allow_tape:
        msg += (
            "\n[files on tape have not been checked, use "
            "allow_tape=True for a complete search]"
        )
    raise ValueError(msg)


def _rank_types(
    match: Mapping[ChannelLike, list[tuple[str, str, float]]],
) -> None:
    """Rank and sort the matched frametypes according to some criteria.

    Parameters
    ----------
    match : `dict` of ``channel: (type, gwf, gapsize)``
        The match dict to sort.
    """
    paths = {typetuple[1] for key in match for typetuple in match[key]}
    rank = {path: (on_tape(path), num_channels(path)) for path in paths}
    # deprioritise types on tape and those with lots of channels
    for key in match:
        match[key].sort(key=lambda x: (-x[2],) + rank[x[1]])


# -- user methods --------------------

# single channel, return_all=False
@overload
def find_frametype(
    channel: str | Channel,
    gpstime: SupportsToGps | None = None,
    *,
    frametype_match: str | re.Pattern | None = None,
    host: str | None = None,
    urltype: str = "file",
    ext: str = "gwf",
    return_all: Literal[False] = False,
    allow_tape: bool = False,
    on_gaps: Literal["error", "ignore", "warn"] = "error",
    **gwdatafind_kw,
) -> str: ...

# single channel, return_all=True
@overload
def find_frametype(
    channel: str | Channel,
    gpstime: SupportsToGps | None = None,
    *,
    frametype_match: str | re.Pattern | None = None,
    host: str | None = None,
    urltype: str = "file",
    ext: str = "gwf",
    return_all: Literal[True] = True,
    allow_tape: bool = False,
    on_gaps: Literal["error", "ignore", "warn"] = "error",
    **gwdatafind_kw,
) -> list[str]: ...

# single channel, return_all not given
@overload
def find_frametype(
    channel: str | Channel,
    gpstime: SupportsToGps | None = None,
    *,
    frametype_match: str | re.Pattern | None = None,
    host: str | None = None,
    urltype: str = "file",
    ext: str = "gwf",
    allow_tape: bool = False,
    on_gaps: Literal["error", "ignore", "warn"] = "error",
    **gwdatafind_kw,
) -> str: ...

# multiple channels, return_all=False
@overload
def find_frametype(
    channel: Iterable[ChannelLike],
    gpstime: SupportsToGps | None = None,
    *,
    frametype_match: str | re.Pattern | None = None,
    host: str | None = None,
    urltype: str = "file",
    ext: str = "gwf",
    return_all: Literal[False] = False,
    allow_tape: bool = False,
    on_gaps: Literal["error", "ignore", "warn"] = "error",
    **gwdatafind_kw,
) -> dict[ChannelLike, str]: ...

# multiple channels, return_all=True
@overload
def find_frametype(
    channel: Iterable[ChannelLike],
    gpstime: SupportsToGps | None = None,
    *,
    frametype_match: str | re.Pattern | None = None,
    host: str | None = None,
    urltype: str = "file",
    ext: str = "gwf",
    return_all: Literal[True] = True,
    allow_tape: bool = False,
    on_gaps: Literal["error", "ignore", "warn"] = "error",
    **gwdatafind_kw,
) -> dict[ChannelLike, list[str]]: ...

# multiple channels, return_all not given
@overload
def find_frametype(
    channel: Iterable[ChannelLike],
    gpstime: SupportsToGps | None = None,
    *,
    frametype_match: str | re.Pattern | None = None,
    host: str | None = None,
    urltype: str = "file",
    ext: str = "gwf",
    allow_tape: bool = False,
    on_gaps: Literal["error", "ignore", "warn"] = "error",
    **gwdatafind_kw,
) -> dict[ChannelLike, list[str]]: ...

def find_frametype(
    channel: str | Channel | Iterable[ChannelLike],
    gpstime: SupportsToGps | None = None,
    *,
    frametype_match: str | re.Pattern | None = None,
    urltype: str = "file",
    ext: str = "gwf",
    return_all: bool = False,
    allow_tape: bool = False,
    cache: bool | None = None,
    on_gaps: Literal["error", "ignore", "warn"] = "error",
    **gwdatafind_kw,
) -> str | list[str] | dict[ChannelLike, str] | dict[ChannelLike, list[str]]:
    """Find the frametype(s) that hold data for a given channel.

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`
        The channel to find.

    gpstime : `int`, optional
        Target GPS time at which to find correct type.

    frametype_match : `str`, optional
        Regular expression to use for frametype `str` matching.

    host : `str`, optional
        Name of datafind host to use.

    urltype : `str`, optional
        The URL type to use.
        Default is "file" to use paths available on the file system.

    ext : `str`, optional
        The file extension for which to search.
        "gwf" is the only file extension supported, but this may be
        extended in the future.

    return_all : `bool`, optional
        If `True` return all found types;
        if `False` (default) return only the 'best' match.

    allow_tape : `bool`, optional
        If `False` (default) do not test types whose frame files are
        stored on tape (not on spinning disk).

    cache : `bool`, `None`, optional
        Whether to cache the contents of remote URLs.
        Default (`None`) is to check the ``GWPY_CACHE`` environment variable.
        See :ref:`gwpy-env-variables` for details.

    on_gaps : `str`, optional
        Action to take when the requested all or some of the GPS interval
        is not covereed by the dataset, one of:

        - ``'error'``: raise a `RuntimeError` (default)
        - ``'warn'``: emit a warning but return all available URLs
        - ``'ignore'``: return the list of URLs with no warnings

    gwdatafind_kw
        Other keyword arguments are passed to the
        `gwdatafind.find_types`, and `gwdatafind.find_urls`.functions.

    Returns
    -------
    If a single name is given, and `return_all=False` (default):

    frametype : `str`
        name of best match frame type

    If a single name is given, and `return_all=True`:

    types : `list` of `str`
        the list of all matching frame types

    If multiple names are given, the above return types are wrapped into a
    `dict[str, str | list[str]]`.

    Examples
    --------
    >>> from gwpy.io import datafind as io_datafind
    >>> io_datafind.find_frametype('H1:IMC-PWR_IN_OUTPUT', gpstime=1126259462)
    'H1_R'
    >>> io_datafind.find_frametype(
    ...     'H1:IMC-PWR_IN_OUTPUT',
    ...     gpstime=1126259462,
    ...     return_all=True,
    ... )
    ['H1_R', 'H1_C']
    >>> io_datafind.find_frametype(
    ...     ('H1:IMC-PWR_IN_OUTPUT', 'H1:OMC-DCPD_SUM_OUTPUT', 'H1:GDS-CALIB_STRAIN'),
    ...     gpstime=1126259462,
    ...     return_all=True,
    ... )
    {'H1:GDS-CALIB_STRAIN': ['H1_HOFT_C00'],
     'H1:OMC-DCPD_SUM_OUTPUT': ['H1_R', 'H1_C'],
     'H1:IMC-PWR_IN_OUTPUT': ['H1_R', 'H1_C']}
    """
    # this function is now horrendously complicated to support a large
    # number of different use cases, hopefully the comments are sufficient

    # check required file extension
    if ext.lower().lstrip(".") != "gwf":
        msg = "Finding frametypes for channels is only supported for ext='gwf'"
        raise ValueError(msg)
    ext = ext.lower().lstrip(".")

    # check required file extension
    if ext.lower().lstrip(".") != "gwf":
        msg = "Finding frametypes for channels is only supported for ext='gwf'"
        raise ValueError(msg)
    ext = ext.lower().lstrip(".")

    # format channel names as list
    if isinstance(channel, list | tuple):
        channels = channel
    else:
        channels = [channel]

    logger.debug("Finding frametype for channels: %s", ",".join(channels))

    # create set() of GWF channel names, and dict map back to user names
    #    this allows users to use nds-style names in this query, e.g.
    #    'X1:TEST.mean,m-trend', and still get results
    chandict: dict[ChannelLike, str] = {c: str(Channel(c).name) for c in channels}
    names = {val: key for key, val in chandict.items()}

    # format GPS time(s)
    if isinstance(gpstime, tuple):
        gpssegment = LigoSegment(*gpstime)
        gpstime = gpssegment[0]
    else:
        gpssegment = None
    if gpstime is not None:
        gpstime = int(to_gps(gpstime))

    # if use gaps post-S5 GPStime, forcibly skip _GRBYYMMDD frametypes at CIT
    if frametype_match is None and gpstime is not None and gpstime > GPS_END_S5:
        frametype_match = NOT_GRB_TYPE

    # -- go

    match: dict[ChannelLike, list[tuple[str, str, float]]] = defaultdict(list)
    searched = set()

    if sess := gwdatafind_kw.pop("session", None):
        ctx = nullcontext(sess)
    elif _gwdatafind_module(**gwdatafind_kw) is gwdatafind:
        ctx = gwdatafind.Session()
    else:
        ctx = nullcontext()

    with ctx as sess:
        if sess:
            gwdatafind_kw["session"] = sess

        for ifo, trend in _parse_ifos_and_trends(channels):
            logger.debug("Finding types for %s", ifo)

            # find all types (prioritising trends if we need to)
            types = find_types(
                ifo,
                match=frametype_match,
                trend=trend,
                ext=ext,
                **gwdatafind_kw,
            )

            logger.debug("Found %s types", len(types))

            # loop over types testing each in turn
            for ftype in types:

                # if we've already search this type for this IFO,
                # don't do it again
                if (ifo, ftype) in searched:
                    continue

                thismatch = _inspect_ftype(
                    list(names),
                    ifo,
                    ftype,
                    gpstime,
                    gpssegment,
                    on_gaps,
                    allow_tape=allow_tape,
                    urltype=urltype,
                    ext=ext,
                    cache=cache,
                    **gwdatafind_kw,
                )

                if thismatch is None:  # failed to read
                    continue

                for name, info in thismatch.items():
                    n = names[name]
                    match[n].append(info)

                    # if only matching once, don't search other types
                    # for this channel
                    if not return_all:
                        names.pop(name)

                # record this type as having been searched
                searched.add((ifo, ftype))

                if not names:  # if all channels matched, stop
                    break

    # raise exception if one or more channels were not found
    _error_missing_channels(
        names.values(),
        match.keys(),
        gpstime,
        allow_tape=allow_tape or urltype != "file",
    )

    # rank types (and pick best if required)
    _rank_types(match)

    # and format as a dict for each channel
    results: dict[ChannelLike, list[str]] = {
        key: list(next(zip(*match[key], strict=True)))
        for key in match
    }

    # -- handle various return scenarios

    # multiple channels, return_all=True
    if isinstance(channel, list | tuple) and return_all:
        return results

    # multiple channels, return_all=False
    if isinstance(channel, list | tuple):
        return {key: val[0] for key, val in results.items()}

    # single channel, return_all=True
    channel = cast("ChannelLike", channel)
    single = results[channel]
    if return_all:
        return single

    # single channel, return_all=False
    return single[0]


def _inspect_ftype(
    names: list[str],
    ifo: str,
    frametype: str,
    gpstime: int | None,
    gpssegment: Segment | None,
    on_gaps: Literal["error", "ignore", "warn"],
    *,
    allow_tape: bool = False,
    cache: bool | None = None,
    **requests_kw,
) -> dict[str, tuple[str, str, float]] | None:
    """Inspect one dataset (frametype) for matches to the required ``names``.

    This function queries GWDataFind for a single file matching this dataset,
    downloads it, and looks at the list of channel names in the TOC.
    """
    logger.debug("Inspecting %s-%s", ifo, frametype)
    # find instance of this frametype
    try:
        path = find_latest(
            ifo,
            frametype,
            gpstime=gpstime,
            allow_tape=allow_tape,
            **requests_kw,
        )
    except (OSError, RuntimeError, IndexError):  # something went wrong
        logger.debug("Failed to find URL for %s-%s", ifo, frametype)
        return None

    # download the file so we can inspect it
    logger.debug("Using URL '%s'", path)
    try:
        path = download_file(path, cache=cache)
    except NETWORK_ERROR as exc:  # failed to download the file
        logger.debug(
            "Failed to download file for %s-%s: %s",
            ifo,
            frametype,
            str(exc),
        )
        return None

    # check for gaps in the record for this type
    gaps = _find_gaps(
        ifo,
        frametype,
        gpssegment,
        on_gaps=on_gaps,
        **requests_kw,
    )

    # record matches for each channel
    match: dict[str, tuple[str, str, float]] = {}

    # search the TOC for one frame file and match any channels
    found = 0
    nchan = len(names)
    try:
        for n in iter_channel_names(path):
            if n in names:  # frametype includes this channel!
                # count how many channels we have found in this type
                found += 1

                # record the match using the user-given channel name
                match[n] = (frametype, path, gaps)

                if found == nchan:  # all channels have been found
                    break

    except (
        OSError,
        RuntimeError,
    ) as exc:  # failed to open file (probably)
        logger.warning(
            "failed to read channels for %s-%s: %s",
            ifo,
            frametype,
            str(exc),
        )
        return None

    logger.debug(
        "Found %s/%s channels in %s-%s",
        len(match),
        nchan,
        ifo,
        frametype,
    )
    return match


@overload
def find_best_frametype(
    channel: str | Channel,
    start: SupportsToGps,
    end: SupportsToGps,
    *,
    allow_tape: bool = True,
    **kwargs,
) -> str:
    ...


@overload
def find_best_frametype(
    channel: Iterable[ChannelLike],
    start: SupportsToGps,
    end: SupportsToGps,
    *,
    allow_tape: bool = True,
    **kwargs,
) -> dict[ChannelLike, str]:
    ...


def find_best_frametype(
    channel: str | Channel | Iterable[ChannelLike],
    start: SupportsToGps,
    end: SupportsToGps,
    *,
    allow_tape: bool = True,
    **kwargs,
) -> str | dict[ChannelLike, str]:
    """Intelligently select the best frametype from which to read this channel.

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

    allow_tape : `bool`, optional
        do not test types whose frame files are stored on tape (not on
        spinning disk)

    kwargs
        Other keyword arguments are passed to `find_frametype`.

    Returns
    -------
    frametype : `str`
        the best matching frametype for the ``channel`` in the
        ``[start, end)`` interval

    Raises
    ------
    ValueError
        If no valid frametypes are found.

    See Also
    --------
    find_frametype
        For details of how available frametypes are matched.

    Examples
    --------
    >>> from gwpy.io.datafind import find_best_frametype
    >>> find_best_frametype('L1:GDS-CALIB_STRAIN', 1126259460, 1126259464)
    'L1_HOFT_C00'
    """
    try:
        # try and find the one true frametype for this channel
        return find_frametype(
            channel,
            gpstime=(start, end),
            allow_tape=allow_tape,
            return_all=False,
            on_gaps="error",
            **kwargs,
        )
    except RuntimeError:  # gaps (or something else went wrong)
        # find all frametypes
        ftout = find_frametype(
            channel,
            gpstime=(start, end),
            allow_tape=allow_tape,
            return_all=True,
            on_gaps="ignore",
            **kwargs,
        )
        # and pick the one with the highest rank
        # (as applied by find_frametype)
        try:
            if isinstance(ftout, dict):
                return {key: ftout[key][0] for key in ftout}
            return ftout[0]
        except IndexError:
            msg = "Cannot find any valid frametypes for channel(s)"
            raise ValueError(msg) from None


@_select_gwdatafind_mod
def find_types(
    observatory: str,
    match: str | re.Pattern | None = None,
    trend: str | None = None,
    **kwargs,
) -> list[str]:
    """Find the available data types for a given observatory.

    Parameters
    ----------
    observatory : `str`
        The observatory for which to search, as a single character,
        e.g ``"G"`` for GEO-600.

    match : `str`, `re.Pattern`, optional
        Regular expression to match types against.

    trend : `str`, optional
        A trend type name to prioritise, e.g. ``"m-trend"``.

    kwargs
        Other arguments are passed to `gwdatafind.find_types`.

    Returns
    -------
    types : `list` of `str`
        A list of matching types, sorted by priority
        (*h(t)* datasets first, large datasets last).

    See Also
    --------
    gwdatafind.find_types
        For details of how available types are discovered.
    """
    types = gwdatafind.find_types(
        observatory,
        match=match,
        **kwargs,
    )
    return sorted(
        types,
        key=partial(_type_priority, trend=trend),
    )


@_select_gwdatafind_mod
def find_urls(
    observatory: str,
    frametype: str,
    start: SupportsToGps,
    end: SupportsToGps,
    on_gaps: Literal["error", "ignore", "warn"] = "error",
    **kwargs,
) -> list[str]:
    """Find the URLs of files of a given data type in a GPS interval.

    Parameters
    ----------
    observatory : `str`
        The observatory for which to search, as a single character,
        e.g ``"G"`` for GEO-600.

    frametype : `str`
        Name of dataset to match.

    start : `~gwpy.time.LIGOTimeGPS`, `int`, `str`
        GPS start time of search.
        any input parseable by `~gwpy.time.to_gps` is fine.
        Non-integer GPS start time will be rounded down.

    end : `~gwpy.time.LIGOTimeGPS`, `int`, `str`
        GPS end time of search.
        any input parseable by `~gwpy.time.to_gps` is fine
        Non-integer GPS end time will be rounded up.

    on_gaps : `str`, optional
        Action to take when the requested all or some of the GPS interval
        is not covereed by the dataset, one of:

        - ``'error'``: raise a `RuntimeError` (default)
        - ``'warn'``: emit a warning but return all available URLs
        - ``'ignore'``: return the list of URLs with no warnings

    kwargs
        Other arguments are passed to `gwdatafind.find_urls`.

    See Also
    --------
    gwdatafind.find_urls
        For details of the underlying discovery tool, and supported
        keyword arguments.
    """
    return gwdatafind.find_urls(
        observatory,
        frametype,
        int(to_gps(start)),
        ceil(to_gps(end)),
        on_gaps=on_gaps,
        **kwargs,
    )


@_select_gwdatafind_mod
def find_latest(
    observatory: str,
    frametype: str,
    gpstime: SupportsToGps | None = None,
    *,
    allow_tape: bool = False,
    **kwargs,
) -> str:
    """Find the path of the latest file of a given data type.

    Parameters
    ----------
    observatory : `str`
        The observatory for which to search, as a single character,
        e.g ``"G"`` for GEO-600.

    frametype : `str`
        Name of dataset to match.

    gpstime : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS time to search for a file,
        any input parseable by `~gwpy.time.to_gps` is fine
        Default (`None`) is to search for the most recent file available.

    allow_tape : `bool`, optional
        If `False` (default) raise `OSError` if the 'latest' URL available
        points to a local path stored on tape (not on spinning disk).

    kwargs
        Other keyword arguments are passed to `gwdatafind.find_latest` or
        `gwdatafind.find_urls`.

    See Also
    --------
    gwdatafind.find_latest
        For details of how the latest URL is discovered and available keyword
        arguments.
        This is called when ``gpstime=None`` is given.

    gwdatafind.find_urls
        For details of how the URLs are discovered when ``gpstime`` is given,
        and available keyword arguments.
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
                on_gaps="ignore",
                **kwargs,
            )[-1]
        else:
            path = gwdatafind.find_latest(
                observatory,
                frametype,
                on_missing="ignore",
                **kwargs,
            )[-1]
    except (
        IndexError,
        RuntimeError,
    ) as exc:
        msg = f"no files found for {observatory}-{frametype}"
        raise RuntimeError(msg) from exc

    if not allow_tape and on_tape(path):
        msg = (
            f"Latest frame file for {observatory}-{frametype} is on tape "
            f"(pass allow_tape=True to force): {path}"
        )
        raise OSError(msg)
    return path
