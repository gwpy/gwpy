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
import typing

from ...detector import ChannelList
from ...io import datafind as io_datafind
from ...time import to_gps
from .. import TimeSeries

if typing.TYPE_CHECKING:
    from re import Pattern

    from ...detector import Channel
    from ...typing import GpsLike
    from .. import (
        TimeSeriesBase,
        TimeSeriesBaseDict,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

logger = logging.getLogger(__name__)


def find(
    channels: list[str | Channel],
    start: GpsLike,
    end: GpsLike,
    *,
    host: str | None = None,
    observatory: str | None = None,
    frametype: str | None = None,
    frametype_match: str | Pattern | None = None,
    pad: float | None = None,
    scaled: bool | None = None,
    allow_tape: bool | None = None,
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

    frametype : `str`, optional
        Name of frametype (dataset) in which this channel is stored.
        Default is to search all available datasets for a match, which
        can be very slow.

    frametype_match : `str`, optional
        Regular expression to use for frametype matching.

    host : `str`, optional
        Name of the GWDataFind server to use.
        Default is set by `gwdatafind.utils.get_default_host`.

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
    start = to_gps(start)
    end = to_gps(end)

    dict_class = series_class.DictClass

    # -- find frametype(s)

    frametypes: dict[str, list[str]] = {}

    if frametype is None:
        logger.debug("Finding frametypes for %d channels", len(channels))
        matched = io_datafind.find_best_frametype(
            channels,
            start,
            end,
            host=host,
            frametype_match=frametype_match,
            allow_tape=allow_tape,
        )

        # flip dict to frametypes with a list of channels
        for name, ftype in matched.items():
            frametypes.setdefault(ftype, []).append(name)

        logger.debug("Determined %s frametypes to read", len(frametypes))
    else:  # use the given frametype for all channels
        frametypes[frametype] = list(map(str, channels))

    # -- read data

    out = dict_class()
    for ftype, clist in frametypes.items():
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
        cache = io_datafind.find_urls(
            obs,
            ftype,
            start,
            end,
            host=host,
            on_gaps="error" if pad is None else "warn",
        )
        if not cache:
            msg = (
                f"No {observatory}-{ftype} URLs found for "
                f"[{start}, {end})"
            )
            raise RuntimeError(msg)

        # read data
        new = dict_class.read(
            cache,
            names,
            start=start,
            end=end,
            pad=pad,
            scaled=scaled,
            parallel=parallel,
            verbose=verbose,
            **readargs,
        )

        # map back to user-given channel name and append
        out.append(type(new)(
            (key, new[chan]) for (key, chan) in zip(clist, names, strict=True)
        ))

    return out
