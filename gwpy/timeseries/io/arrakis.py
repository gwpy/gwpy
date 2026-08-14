# Copyright (c) 2026 Cardiff University
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

"""Fetch data from Arrakis."""

from __future__ import annotations

import logging
import os
from math import ceil
from typing import TYPE_CHECKING

from ...time import to_gps
from .. import (
    StateVector,
    StateVectorDict,
    TimeSeries,
    TimeSeriesDict,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )
    from typing import (
        Any,
        TypeVar,
    )

    import arrakis

    from ...detector import Channel
    from ...time import SupportsToGps
    from .. import (
        TimeSeriesBase,
        TimeSeriesBaseDict,
    )

    _T = TypeVar("_T", bound=TimeSeriesBase)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

logger = logging.getLogger(__name__)


def fetch_series(
    channel: str | Channel,
    start: SupportsToGps,
    end: SupportsToGps,
    *,
    client: arrakis.Client | None = None,
    url: str | None = None,
    pad: float | None = None,
    series_class: type[_T] = TimeSeries,
    **kwargs,
) -> _T:
    """Fetch a single data series from Arrakis.

    channel : `str`, `~gwpy.detector.Channel`
        The name (or representation) of the data channel to fetch.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    client : `arrakis.Client`, optional
        The active Arrakis client to use.

    url : `str`, optional
        URL of Arrakis server to use, if blank will try the
        ``ARRAKIS_SERVER`` environment variable, if set.

    pad : `float`, `int`, optional
        Value to insert between gaps.
        The given type is automatically cast to the array's dtype,
        so look out for precision loss if you use a float value
        with an integer array.
        Default behaviour is to raise an exception when any gaps are
        found.

    series_class : `type`, optional
        The type to use for each `Series` instance.
        Default set by the class object used to call ``.get()``.

    kwargs
        Other keyword arguments to pass to `arrakis.Client.fetch()`.

    Returns
    -------
    data : `TimeSeries` or `StateVector`
        A new `TimeSeries` or `StateVector` fetched from Arrakis.
    """
    return fetch_block(
        [channel],
        start,
        end,
        client=client,
        url=url,
        series_class=series_class,
        pad=pad,
        **kwargs,
    )[str(channel)]


def fetch_block(
    channels: Sequence[str | Channel],
    start: SupportsToGps,
    end: SupportsToGps,
    *,
    client: arrakis.Client | None = None,
    url: str | None = None,
    pad: float | None = None,
    series_class: type[_T] = TimeSeries,
    **kwargs,
) -> TimeSeriesBaseDict[_T]:
    """Fetch a dict of series data from Arrakis.

    Parameters
    ----------
    channels : `list` of `str` or `Channel`
        List of channel names to fetch.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    client : `arrakis.Client`, optional
        The active Arrakis client to use.

    url : `str`, optional
        URL of Arrakis server to use, if blank will try the
        ``ARRAKIS_SERVER`` environment variable, if set.

    pad : `float`, optional
        Float value to insert between gaps.
        Default behaviour is to raise an exception when any gaps are
        found.

    series_class : `type`, optional
        The type to use for each `Series` instance.
        Default set by the class object used to call ``.get()``.

    kwargs
        Other keyword arguments to pass to `arrakis.Client.fetch()`.

    Returns
    -------
    data : `TimeSeriesBaseDict` or similar
        A new structured `dict` (e.g. `TimeSeriesDict`) of
        `(str, TimeSeries)` pairs fetched from Arrakis.
    """
    import arrakis
    import arrakis.client

    # format GPS times
    gpsstart = float(to_gps(start))
    gpsend = float(to_gps(end))

    # read using integers
    readstart = int(gpsstart)
    readend = ceil(gpsend)

    # map channels to strings
    names = {str(name): name for name in channels}

    # get an Arrakis client connection
    if client:
        logger.debug("Using existing Arrakis connection (%s)", client.initial_url)
    else:
        client = arrakis.Client(url)
        logger.debug("Connected to Arrakis server at %s", client.initial_url)

    # configure gap handling
    kwargs.setdefault(
        "on_gap",
        "raise" if pad is None else arrakis.client.ONGAP_DEFAULT,
    )

    # fetch data
    logger.debug("Fetching data for [%s, %s)...", readstart, readend)
    block = client.fetch(
        list(names),
        readstart,
        readend,
        **kwargs,
    )

    # transform into our type, handling padding
    out = series_class.DictClass()
    for name, series in block.items():
        if series.has_gaps:
            series.data.set_fill_value(pad)  # ty:ignore[unresolved-attribute]
        # Use original key (maybe a Channel object)
        out[names[name]] = series_class.from_arrakis(series)

    # constrain to original request times
    if gpsstart != readstart or gpsend != readend:
        out.crop(start=gpsstart, end=gpsend)

    return out


# -- get registry --------------------

def identify_sources(
    origin: str,
    *args: Any,  # noqa: ARG001
    client: arrakis.Client | None = None,
    url: str | None = None,
    **kwargs,  # noqa: ARG001
) -> Iterable[dict[str, object]] | None:
    """Identify Arrakis sources for these arguments."""
    # Arrakis only works for 'get'
    if origin != "get":
        return None

    # We must have the arrakis client available
    try:
        from arrakis.client import (
            Client,
            parse_arrakis_url,
        )
    except ImportError:
        return None

    # If given an arrakis client, use it
    if isinstance(client, Client):
        return [{
            "client": client,
            "priority": 1,
        }]

    # Otherwise try and get the URL of the configured Arrakis server
    if url is None and not (url := os.getenv("ARRAKIS_SERVER")):
        return []

    return [{
        "url": parse_arrakis_url(url).geturl(),
        "priority": 10,
    }]


for klass, fetch in (
    (TimeSeries, fetch_series),
    (StateVector, fetch_series),
    (TimeSeriesDict, fetch_block),
    (StateVectorDict, fetch_block),
):
    klass.get.registry.register_identifier(
        "arrakis",
        klass,
        identify_sources,
    )
    klass.get.registry.register_reader(
        "arrakis",
        klass,
        fetch,
    )
