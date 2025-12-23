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

"""GWDatafind I/O integration for `gwpy.timeseries`."""

from __future__ import annotations

import logging
from itertools import product
from typing import TYPE_CHECKING

from gwdatafind.utils import get_default_host

from ...detector import (
    Channel,
    ChannelList,
)
from ...io import datafind as io_datafind
from ...time import to_gps
from .. import (
    StateVector,
    StateVectorDict,
    TimeSeries,
    TimeSeriesDict,
)
from .losc import _any_gwosc_channels

if TYPE_CHECKING:
    from collections.abc import (
        Collection,
        Iterable,
    )
    from re import Pattern

    from ...time import SupportsToGps
    from .. import (
        TimeSeriesBase,
        TimeSeriesBaseDict,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

try:
    GWDATAFIND_SERVER = get_default_host()
except ValueError:
    GWDATAFIND_SERVER = None

DEFAULT_GWDATAFIND_SERVERS = {
    "local": GWDATAFIND_SERVER,
    "igwn": "datafind.igwn.org",
    "gwosc": "datafind.gwosc.org",
}

logger = logging.getLogger(__name__)


def find_series(
    channel: str | Channel,
    start: SupportsToGps,
    end: SupportsToGps,
    *,
    observatory: str | None = None,
    frametype: str | None = None,
    frametype_match: str | Pattern | None = None,
    host: str | None = None,
    urltype: str | None = "file",
    ext: str = "gwf",
    pad: float | None = None,
    scaled: bool | None = None,
    allow_tape: bool | None = None,
    parallel: int = 1,
    verbose: bool | str = False,
    series_class: type[TimeSeriesBase] = TimeSeries,
    **readargs,
) -> TimeSeriesBase:
    """Find and return data for a single channel using GWDataFind.

    This method uses :mod:`gwdatafind` to discover the URLs
    that provide the requested data, then reads those files using
    :meth:`TimeSeries.read()`.

    Parameters
    ----------
    channel : `str`, `Channel`
        Name of data channel to find.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    observatory : `str`, optional
        The observatory to use when searching for data.
        Default is to use the observatory from the channel name prefix,
        but this should be specified when searching for data in a
        multi-observatory dataset (e.g. `observatory='HLV'`).

    frametype : `str`, optional
        Name of frametype (dataset) in which this channel is stored.
        Default is to search all available datasets for a match, which
        can be very slow.

    frametype_match : `str`, optional
        Regular expression to use for frametype matching.

    host : `str`, optional
        Default is set by `gwdatafind.utils.get_default_host`.

    urltype : `str`, optional
        The URL type to use.
        Default is "file" to use paths available on the file system.

    ext : `str`, optional
        The file extension for which to search.
        "gwf" is the only file extension supported, but this may be
        extended in the future.

    pad : `float`, optional
        Value with which to fill gaps in the source data,
        by default gaps will result in a `ValueError`.

    scaled : `bool`, optional
        Apply slope and bias calibration to ADC data, for non-ADC data
        this option has no effect.

    parallel : `int`, optional
        Number of parallel threads to use when reading data.

    allow_tape : `bool`, optional
        Allow reading from frame files on (slow) magnetic tape.

    series_class : `type`, optional
        The type to use for each `Series` instance.
        Default is `TimeSeries`.

    verbose: `bool`, optional
        Use debug-level logging for data access progress.
        If ``verbose`` is specified as a string, this also defines
        the prefix for the progress meter when reading data.

    readargs
        Any other keyword arguments to be passed to `.read()`.

    Raises
    ------
    requests.exceptions.HTTPError
        If the GWDataFind query fails for any reason.

    RuntimeError
        If no files are found to read, or if the read operation
        fails.
    """
    return find_dict(
        [channel],
        start,
        end,
        observatory=observatory,
        frametype=frametype,
        frametype_match=frametype_match,
        host=host,
        urltype=urltype,
        ext=ext,
        pad=pad,
        scaled=scaled,
        allow_tape=allow_tape,
        parallel=parallel,
        verbose=verbose,
        series_class=series_class,
        **readargs,
    )[channel]


def find_dict(
    channels: Collection[str | Channel],
    start: SupportsToGps,
    end: SupportsToGps,
    *,
    observatory: str | None = None,
    frametype: str | dict[str | Channel, str | None] | None = None,
    frametype_match: str | Pattern | None = None,
    host: str | None = None,
    urltype: str | None = "file",
    ext: str = "gwf",
    pad: float | None = None,
    scaled: bool | None = None,
    allow_tape: bool | None = None,
    cache: bool | None = None,
    parallel: int = 1,
    verbose: bool | str = False,
    series_class: type[TimeSeriesBase] = TimeSeries,
    **readargs,
) -> TimeSeriesBaseDict:
    """Find and return data for multiple channels using GWDataFind.

    This method uses :mod:`gwdatafind` to discover the URLs
    that provide the requested data, then reads those files using
    :meth:`TimeSeriesDict.read()`.

    Parameters
    ----------
    channels : `list`
        List of names of data channels to find.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    observatory : `str`, optional
        The observatory to use when searching for data.
        Default is to use the observatory from the channel name prefix,
        but this should be specified when searching for data in a
        multi-observatory dataset (e.g. `observatory='HLV'`).

    frametype : `str`, `dict`, optional
        Name of frametype (dataset) in which this channel is stored.
        Default is to search all available datasets for a match, which
        can be very slow.
        A `dict` can be provided mapping each input channel to a specific
        frametype, with `None` (or missing) values indicating channels
        for which the frametype should be determined automatically.

    frametype_match : `str`, optional
        Regular expression to use for frametype matching.

    host : `str`, optional
        Name of the GWDataFind server to use.
        Default is set by `gwdatafind.utils.get_default_host`.

    urltype : `str`, optional
        The URL type to use.
        Default is "file" to use paths available on the file system.

    ext : `str`, optional
        The file extension for which to search.
        "gwf" is the only file extension supported, but this may be
        extended in the future.

    pad : `float`, optional
        Value with which to fill gaps in the source data,
        by default gaps will result in a `ValueError`.

    scaled : `bool`, optional
        Apply slope and bias calibration to ADC data, for non-ADC data
        this option has no effect.

    parallel : `int`, optional
        Number of parallel threads to use when reading data.

    allow_tape : `bool`, optional
        Allow reading from frame files on (slow) magnetic tape.

    cache : `bool`, `None`, optional
        Whether to cache the contents of remote URLs.
        Default (`None`) is to check the ``GWPY_CACHE`` environment variable.
        See :ref:`gwpy-env-variables` for details.

    series_class : `type`, optional
        The type to use for each `Series` instance.
        Default is `TimeSeries`.

    verbose: `bool`, optional
        Use debug-level logging for data access progress.
        If ``verbose`` is specified as a string, this also defines
        the prefix for the progress meter when reading data.

    readargs
        Any other keyword arguments to be passed to `.read()`.

    Raises
    ------
    requests.exceptions.HTTPError
        If the GWDataFind query fails for any reason.

    RuntimeError
        If no files are found to read, or if the read operation
        fails.
    """
    start = to_gps(start)
    end = to_gps(end)

    dict_class = series_class.DictClass

    # -- find frametype(s)

    groups: dict[str, list[str]] = {}
    if isinstance(frametype, dict):
        frametypes = frametype
    else:
        frametypes = dict.fromkeys(channels, frametype)
    missing = [c for c in channels if not frametypes.get(c)]

    if missing:
        logger.debug("Finding frametypes for %d channels", len(missing))
        frametypes |= io_datafind.find_best_frametype(
            missing,
            start,
            end,
            host=host,
            ext=ext,
            urltype=urltype,
            frametype_match=frametype_match,
            allow_tape=bool(allow_tape),
            cache=cache,
        )

    # flip dict to frametypes with a list of channels
    for name, ftype in frametypes.items():
        if ftype is None:
            msg = f"Cannot determine frametype for channel '{name}'"
            raise RuntimeError(msg)
        groups.setdefault(ftype, []).append(name)  # type: ignore[arg-type]

    logger.debug("Determined %s frametypes to read", len(groups))

    # -- read data

    out = dict_class()
    for ftype, clist in groups.items():
        logger.debug(
            "Reading %d channels from '%s': %s",
            len(clist),
            ftype,
            ", ".join(clist),
        )

        # parse as a ChannelList
        channellist = ChannelList.from_names(*clist)
        # strip trend tags from channel names
        names = [c.name for c in channellist]

        # find observatory for this group
        if observatory is None:
            try:
                obs = "".join(
                    sorted({c.ifo[0] for c in channellist}),
                )
            except TypeError as exc:
                msg = "Cannot parse list of IFOs from channel names"
                raise ValueError(msg) from exc

        else:
            obs = observatory

        # find frames
        urls = io_datafind.find_urls(
            obs,
            ftype,
            start,
            end,
            host=host,
            ext=ext,
            urltype=urltype,
            on_gaps="error" if pad is None else "warn",
        )
        if not urls:
            msg = (
                f"No {observatory}-{ftype} URLs found for "
                f"[{start}, {end})"
            )
            raise RuntimeError(msg)

        # read data
        new = dict_class.read(
            urls,
            names,
            start=start,
            end=end,
            pad=pad,
            scaled=scaled,
            parallel=parallel,
            verbose=verbose,
            cache=cache,
            **readargs,
        )

        # map back to user-given channel name and append
        out.append(type(new)(
            (key, new[chan]) for (key, chan) in zip(clist, names, strict=True)
        ))

    return out


# -- get registry --------------------

def _as_tuple(
    value: str | Iterable[str] | None,
    default: Iterable[str],
) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def identify_gwdatafind_sources(
    origin: str,
    channels: str | Channel | Iterable[str | Channel],
    *args,  # noqa: ARG001
    host: str | None = None,
    urltype: str | None = None,
    ext: str | None = None,
    **kwargs,  # noqa: ARG001
) -> Iterable[dict[str, object]] | None:
    """Identify GWDataFind sources and options based on the input arguments.

    This function is registered with the `get` registry for the relevant
    timeseries classes, and is called to determine what gwdatafind endpoints
    and options can be used to get the requested data.
    """
    # If not called for "get", then not applicable
    if origin != "get":
        return None

    # If the host is an NDS server, don't get involved
    if str(host).startswith("nds"):
        return []

    # Make channels a list
    channels = (channels,) if isinstance(channels, str | Channel) else channels

    # Just IFO names, not channels, most likely a GWOSC request
    if all(len(str(c)) <= 2 for c in channels):
        return []

    # Now iterate over possible gwdatafind sources by combining known options
    # for host, urltype, and extension:

    sources: list[dict[str, object]] = []

    gwosc = _any_gwosc_channels(channels)
    hosts = _as_tuple(host, filter(None, DEFAULT_GWDATAFIND_SERVERS.values()))
    urltypes = _as_tuple(urltype, ("file", "osdf"))
    exts = _as_tuple(ext, ("gwf",))
    for host_, urltype_, ext_ in product(
        hosts,
        urltypes,
        exts,
    ):
        if gwosc and host_ == DEFAULT_GWDATAFIND_SERVERS["gwosc"]:
            # Prioritise the GWOSC server for GWOSC channels
            priority = 5
        else:
            priority = 10
        sources.append({
            "host": host_,
            "urltype": urltype_,
            "ext": ext_,
            "priority": priority,
        })

    return sources


for klass, find in (
    (TimeSeries, find_series),
    (TimeSeriesDict, find_dict),
    (StateVector, find_series),
    (StateVectorDict, find_dict),
):
    klass.get.registry.register_identifier(
        "gwdatafind",
        klass,
        identify_gwdatafind_sources,
    )
    klass.get.registry.register_reader(
        "gwdatafind",
        klass,
        find,
    )
