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

"""NDS2 data query routines for the TimeSeries."""

from __future__ import annotations

import importlib.util
import logging
import operator
from functools import reduce
from math import ceil
from typing import TYPE_CHECKING

from numpy import ones as numpy_ones

from ...detector import Channel
from ...io import nds2 as io_nds2
from ...segments import (
    Segment,
    SegmentList,
)
from ...time import to_gps
from ...utils.progress import progress_bar
from .. import (
    StateVector,
    StateVectorDict,
    TimeSeries,
    TimeSeriesDict,
)
from ..connect import _pad_series
from .losc import _any_gwosc_channels

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )
    from typing import (
        SupportsFloat,
        TypeVar,
    )

    import nds2

    from ...time import SupportsToGps
    from .. import (
        TimeSeriesBase,
        TimeSeriesBaseDict,
    )

    _TChan = TypeVar("_TChan")
    _T = TypeVar("_T", bound=TimeSeriesBase)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

logger = logging.getLogger(__name__)


def log_nds2(
    connection: nds2.connection | str,
    message: str,
    *args: object,
    level: int = logging.DEBUG,
    **kwargs,
) -> None:
    """Emit a log message related to an open NDS2 connection."""
    if not isinstance(connection, str):
        connection = connection.get_host()
    message = f"[{connection}] {message}"
    logger.log(level, message, *args, **kwargs)


def _parse_nds_enum_dict_param(
    channels: Iterable[_TChan],
    key: str,
    value: dict | int | str | None,
) -> dict[_TChan, int]:
    """Parse an NDS2 Enum value into a dict keyed on ``channels``."""
    # parse input key
    enum: type[io_nds2._Nds2Enum]
    if key == "type":
        enum = io_nds2.Nds2ChannelType
        default = enum.any().value
    else:
        enum = io_nds2.Nds2DataType
        default = enum.any().value

    # set default
    if value is None:
        value = default

    # parse non-int enum representation
    if not isinstance(value, dict | int):
        value = enum.find(value).value

    # return dict of ints
    if isinstance(value, int):
        return dict.fromkeys(channels, value)

    # here we know ``value`` is a dict, so just fill in the blanks
    value = value.copy()
    for chan in channels:
        value.setdefault(chan, default)
    return value


def _set_parameter(
    connection: nds2.connection,
    parameter: str,
    value: str | int | bool,  # noqa: FBT001
) -> None:
    """Set a parameter for the connection, handling errors as warnings."""
    ref = f"{parameter}='{value}'"
    if connection.set_parameter(parameter, str(value)):
        log_nds2(connection, "Set %s", ref)
    else:
        log_nds2(connection, "Failed to set %s", ref, level=logging.WARNING)



def fetch_series(
    channel: str | Channel,
    start: SupportsToGps,
    end: SupportsToGps,
    *,
    type: str | int | dict | None = None,  # noqa: A002
    dtype: str | int | dict | None = None,
    allow_tape: bool | None = None,
    connection: nds2.connection | None = None,
    host: str | None = None,
    port: int | None = None,
    pad: float | None = None,
    scaled: bool | None = None,
    series_class: type[_T] = TimeSeries,
    verbose: bool | str = False,
) -> _T:
    """Fetch a single data series from NDS2.

    channel : `str`, `~gwpy.detector.Channel`
        The name (or representation) of the data channel to fetch.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    host : `str`, optional
        URL of NDS server to use, if blank will try any server
        (in a relatively sensible order) to get the data

        One of ``connection`` or ``host`` must be given.

    port : `int`, optional
        Port number for NDS server query, must be given with `host`.

    verify : `bool`, optional
        Check channels exist in database before asking for data.
        Default is `True`.

    verbose : `bool`, optional
        This argument is deprecated and will be removed in a future release.
        Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

    connection : `nds2.connection`, optional
        Open NDS connection to use.
        Default is to open a new connection using ``host`` and ``port``
        arguments.

        One of ``connection`` or ``host`` must be given.

    pad : `float`, optional
        Float value to insert between gaps.
        Default behaviour is to raise an exception when any gaps are
        found.

    scaled : `bool`, optional
        Apply slope and bias calibration to ADC data, for non-ADC data
        this option has no effect.

    allow_tape : `bool`, optional
        Allow data access from slow tapes.
        If ``host`` or ``connection`` is given, the default is to do
        whatever the server default is, otherwise servers will be searched
        with ``allow_tape=False`` first, then ``allow_tape=True`` if that
        fails.

    type : `int`, `str`, optional
        NDS2 channel type integer or string name to match.
        Default is to search for any channel type.

    dtype : `numpy.dtype`, `str`, `type`, or `dict`, optional
        NDS2 data type to match.
        Default is to search for any data type.

    Returns
    -------
    data : `TimeSeries` or `StateVector`
        A new `TimeSeries` or `StateVector` fetched from NDS.
    """
    return fetch_dict(
        [channel],
        start,
        end,
        type=type,
        dtype=dtype,
        allow_tape=allow_tape,
        connection=connection,
        host=host,
        port=port,
        pad=pad,
        scaled=scaled,
        series_class=series_class,
        verbose=verbose,
    )[str(channel)]


def fetch_dict(
    channels: Sequence[str | Channel],
    start: SupportsToGps,
    end: SupportsToGps,
    *,
    type: str | int | dict | None = None,  # noqa: A002
    dtype: str | int | dict | None = None,
    allow_tape: bool | None = None,
    connection: nds2.connection | None = None,
    host: str | None = None,
    port: int | None = None,
    pad: float | None = None,
    scaled: bool | None = None,
    series_class: type[_T] = TimeSeries,
    verbose: bool | str = False,
) -> TimeSeriesBaseDict[_T]:
    """Fetch a dict of series data from NDS2.

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

    host : `str`, optional
        URL of NDS server to use, if blank will try any server
        (in a relatively sensible order) to get the data

        One of ``connection`` or ``host`` must be given.

    port : `int`, optional
        Port number for NDS server query, must be given with `host`.

    verify : `bool`, optional
        Check channels exist in database before asking for data.
        Default is `True`.

    verbose : `bool`, optional
        This argument is deprecated and will be removed in a future release.
        Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

    connection : `nds2.connection`, optional
        Open NDS connection to use.
        Default is to open a new connection using ``host`` and ``port``
        arguments.

        One of ``connection`` or ``host`` must be given.

    pad : `float`, optional
        Float value to insert between gaps.
        Default behaviour is to raise an exception when any gaps are
        found.

    scaled : `bool`, optional
        Apply slope and bias calibration to ADC data, for non-ADC data
        this option has no effect.

    allow_tape : `bool`, optional
        Allow data access from slow tapes.
        If ``host`` or ``connection`` is given, the default is to do
        whatever the server default is, otherwise servers will be searched
        with ``allow_tape=False`` first, then ``allow_tape=True`` if that
        fails.

    type : `int`, `str`, optional
        NDS2 channel type integer or string name to match.
        Default is to search for any channel type.

    dtype : `numpy.dtype`, `str`, `type`, or `dict`, optional
        NDS2 data type to match.
        Default is to search for any data type.

    Returns
    -------
    data : `TimeSeriesBaseDict`
        A new `TimeSeriesBaseDict` of (`str`, `TimeSeries`) pairs fetched
        from NDS.
    """
    # format GPS times
    gpsstart = to_gps(start)
    gpsend = to_gps(end)

    # format verbose progress
    desc: str
    if isinstance(verbose, str):
        desc = verbose
        verbose = True
    else:
        desc = "Downloading data"
        verbose = bool(verbose)

    # get a connection to an NDS2 server
    if connection:
        log_nds2(connection, "Using existing connection")
    else:
        if host is None:
            msg = (
                "either connection= or host= keyword arguments "
                "must be given to fetch data from NDS2"
            )
            raise ValueError(msg)
        connection = io_nds2.auth_connect(host, port)

    # set ALLOW_DATA_ON_TAPE
    if allow_tape is not None:
        _set_parameter(
            connection,
            "ALLOW_DATA_ON_TAPE",
            str(allow_tape),
        )

    ctype = _parse_nds_enum_dict_param(channels, "type", type)
    dtype = _parse_nds_enum_dict_param(channels, "dtype", dtype)

    # read using integers
    istart = int(gpsstart)
    iend = ceil(gpsend)

    # verify channels exist
    log_nds2(connection, "Checking channels list against database")
    utype = reduce(operator.or_, ctype.values())  # logical OR of types
    udtype = reduce(operator.or_, dtype.values())
    epoch = (istart, iend) if connection.get_protocol() > 1 else None
    ndschannels = io_nds2.find_channels(
        channels,
        connection=connection,
        epoch=epoch,
        type=utype,
        dtype=udtype,
        unique=True,
    )

    names = [Channel.from_nds2(c).ndsname for c in ndschannels]
    log_nds2(connection, "Channel check complete, %s names found", len(names))

    # handle minute trend timing
    if any(c.endswith("m-trend") for c in names) and (istart % 60 or iend % 60):
        log_nds2(
            connection,
            "Requested at least one minute trend, but "
            "start and stop GPS times are not multiples of "
            "60; times will be expanded outwards to compensate",
        )
        istart, iend = io_nds2.minute_trend_times(istart, iend)

    # get data availability
    span = SegmentList([Segment(istart, iend)])
    if pad is None:
        qsegs = span
        gap = "raise"
    elif connection.get_protocol() == 1:
        qsegs = span
        gap = "pad"
    else:
        log_nds2(connection, "Querying for data availability...")
        pad = float(pad)
        gap = "pad"
        qsegs = _get_data_segments(ndschannels, istart, iend, connection) & span
        log_nds2(
            connection,
            "Availability check complete, found %s viable segments of data "
            "with %.2f%% coverage",
            len(qsegs),
            abs(qsegs) / abs(span) * 100,
        )
        if span - qsegs:
            log_nds2(connection, "Gaps will be padded with %s", pad)

    # query for each segment
    out = series_class.DictClass()
    with progress_bar(
        total=float(abs(qsegs)),
        desc=desc,
        unit="s",
        disable=not bool(verbose),
    ) as bar:
        for seg in qsegs:
            total = 0.
            for buffers in connection.iterate(int(seg[0]), int(seg[1]), names):
                for buffer_, chan in zip(buffers, channels, strict=True):
                    series = series_class.from_nds2_buffer(
                        buffer_,
                        scaled=scaled,
                        copy=chan not in out,  # only copy if first buffer
                    )
                    out.append({chan: series}, pad=pad, gap=gap)
                new = buffer_.length / buffer_.channel.sample_rate
                total += new
                bar.update(new)
            # sometimes NDS2 returns no data at all
            if not total and gap != "pad":
                msg = f"no data received from {connection.get_host()} for {seg}"
                raise RuntimeError(msg)

    # finalise timeseries to make sure each channel has the correct limits
    # only if user asked to pad gaps
    if pad is not None:
        for chan, ndschan in zip(channels, ndschannels, strict=True):
            try:
                ts = out[chan]
            except KeyError:
                out[chan] = _create_series(
                    ndschan,
                    pad,
                    gpsstart,
                    gpsend,
                    series_class=series_class,
                )
            else:
                out[chan] = _pad_series(ts, pad, gpsstart, gpsend)

    # constrain to the non-integer GPS times we were actually given
    if gpsstart != istart or gpsend != iend:
        out.crop(start=gpsstart, end=gpsend)

    return out


def _create_series(
    ndschan: nds2.channel,
    value: float,
    start: SupportsFloat,
    end: SupportsFloat,
    series_class: type[TimeSeriesBase] = TimeSeries,
) -> TimeSeriesBase:
    """Create a timeseries to cover the specified [start, end) limits.

    To cover a gap in data returned from NDS.
    """
    channel = Channel.from_nds2(ndschan)
    nsamp = int((end - start) * channel.sample_rate.value)  # type: ignore[union-attr]
    return series_class(
        numpy_ones(nsamp) * value,
        t0=start,
        sample_rate=channel.sample_rate,
        unit=channel.unit,
        channel=channel)


def _get_data_segments(
    channels: list[nds2.channel],
    start: int,
    end: int,
    connection: nds2.connection,
) -> SegmentList:
    """Get available data segments for the given channels."""
    allsegs = io_nds2.get_availability(
        channels,
        start,
        end,
        connection=connection,
    )
    return allsegs.intersection(allsegs.keys())


# -- get registry --------------------


def identify_nds2_sources(
    origin: str,
    channels: str | Channel | Iterable[str | Channel],
    start: SupportsToGps | None,
    *args,  # noqa: ARG001
    connection: nds2.connection | None = None,
    host: str | None = None,
    port: int | None = None,
    allow_tape: bool | None = None,
    **kwargs,  # noqa: ARG001
) -> Iterable[dict[str, object]] | None:
    """Identify NDS2 sources for these arguments."""
    # NDS2 only works for 'get'
    if origin != "get":
        return None

    # We must be able to import the NDS2 client
    if importlib.util.find_spec("nds2") is None:
        return None

    # If the host is a GWDataFind NDS server, don't get involved
    if "datafind" in str(host):
        return []

    if connection is not None or host is not None:
        return [{
            "connection": connection,
            "host": host,
            "port": port,
            "allow_tape": allow_tape,
        }]

    channels = (channels,) if isinstance(channels, (str, Channel)) else channels

    # Just IFO names, not channels, most likely a GWOSC request
    if all(len(str(c)) <= 2 for c in channels):
        return []

    # Now iterate over possible NDS2 sources by combining known options
    # for host/port and allow_tape:

    sources: list[dict[str, object]] = []

    ifos = {Channel(channel).ifo for channel in channels}
    try:
        ifo = ifos.pop()
    except KeyError:  # ifos is empty
        ifo = None

    hostlist = io_nds2.host_resolution_order(ifo, epoch=start)

    if allow_tape is None:
        tapes = [False, True]
    else:
        tapes = [allow_tape]

    gwosc = _any_gwosc_channels(channels)

    for allow_tape_ in tapes:
        for host_, port_ in hostlist:
            # Deprioritise tape-allowed connections
            priority = 100 if allow_tape_ else 10
            if gwosc and "gwosc" in host_:
                # Prioritise the GWOSC server for GWOSC channels
                priority -= 1
            sources.append({
                "connection": None,
                "host": host_,
                "port": port_,
                "allow_tape": allow_tape_,
                "priority": priority,
            })

    return sources


for klass, fetch in (
    (TimeSeries, fetch_series),
    (StateVector, fetch_series),
    (TimeSeriesDict, fetch_dict),
    (StateVectorDict, fetch_dict),
):
    klass.get.registry.register_identifier(
        "nds2",
        klass,
        identify_nds2_sources,
    )
    klass.get.registry.register_reader(
        "nds2",
        klass,
        fetch,
    )
