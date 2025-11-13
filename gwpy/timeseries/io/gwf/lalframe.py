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

"""Read gravitational-wave frame (GWF) files using the LALFrame API.

The frame format is defined in :dcc:`LIGO-T970130`.
"""

from __future__ import annotations

import contextlib
import os.path
import warnings
from typing import TYPE_CHECKING

import lal
import lalframe

from ....io.gwf import lalframe as io_gwf_lalframe
from ....io.utils import (
    file_list,
    file_path,
)
from ....segments import Segment
from ....utils import lal as lalutils
from ... import TimeSeries

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

    from ....time import LIGOTimeGPS
    from ... import (
        TimeSeriesBase,
        TimeSeriesBaseDict,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- utilities -----------------------

def open_data_source(
    source: str | Path | IO | lal.Cache | list[str | Path | IO],
) -> lalframe.FrStream:
    """Open a GWF file source into a |lalframe.FrStream|_ object.

    Parameters
    ----------
    source : `str`, `file`, `list`
        Data source to read.

    Returns
    -------
    stream : |lalframe.FrStream|_
        An open `lalframe.FrStream`.

    Raises
    ------
    ValueError
        If the input format cannot be identified.
    """
    # -- preformatting

    with contextlib.suppress(ValueError):
        source = file_path(source)

    # import cache from file
    if (
        isinstance(source, str)
        and source.endswith((".lcf", ".cache"))
    ):
        source = lal.CacheImport(source)

    # reformat cache (or any list of files) as a lal cache object
    if isinstance(source, list):
        cache = lal.Cache()
        for entry in file_list(source):
            cache = lal.CacheMerge(cache, lal.CacheGlob(*os.path.split(entry)))
        source = cache

    # -- now we have a lal.Cache or a filename

    # read lal cache object
    if isinstance(source, lal.Cache):
        return lalframe.FrStreamCacheOpen(source)

    # read single file
    if isinstance(source, str):
        return lalframe.FrStreamOpen(*map(str, os.path.split(source)))

    msg = f"Don't know how to open data source of type '{type(source).__name__}'"
    raise ValueError(msg)


def get_stream_segment(stream: lalframe.FrStream) -> Segment:
    """Get the GPS end time of the last frame in this stream.

    Parameters
    ----------
    stream : |lalframe.FrStream|_
        Stream of data to search.

    Returns
    -------
    segment: `gwpy.segments.Segment`
        The GPS ``[start, end)`` segment of data covered by this stream.
    """
    epoch = lal.LIGOTimeGPS(stream.epoch)
    try:
        # rewind stream to the start
        lalframe.FrStreamRewind(stream)
        start = lal.LIGOTimeGPS(stream.epoch)
        end = 0.
        # loop over each file in the stream cache and query its duration
        for _ in range(stream.cache.length):
            for j in range(lalframe.FrFileQueryNFrame(stream.file)):
                # get GPS start and duration of this frame
                fstart = lalframe.FrFileQueryGTime(
                    lal.LIGOTimeGPS(),
                    stream.file,
                    j,
                )
                dt = lalframe.FrFileQueryDt(stream.file, j)
                # record the end time of the last frame
                end = max(end, fstart + dt)
                # move on to the next frame
                lalframe.FrStreamNext(stream)
        return Segment(start, end)
    finally:
        # reset stream to where we started and return
        lalframe.FrStreamSeek(stream, epoch)


def get_stream_duration(stream: lalframe.FrStream) -> float:
    """Calculate the duration of time stored in a frame stream.

    Parameters
    ----------
    stream : |lalframe.FrStream|_
        Stream of data to search.

    Returns
    -------
    duration : `float`
        The duration (seconds) of the data for this channel.
    """
    return abs(get_stream_segment(stream))


# -- read ----------------------------

def read(
    source: str | Path | IO | list[str | Path | IO],
    channels: list[str],
    start: LIGOTimeGPS | None = None,
    end: LIGOTimeGPS | None = None,
    type: str | dict[str, str] | None = None,
    series_class: type[TimeSeriesBase] = TimeSeries,
    scaled: bool | None = None,
) -> TimeSeriesBaseDict:
    """Read data from one or more GWF files using the LALFrame API."""
    # scaled must be provided to provide a consistent API with frameCPP
    if scaled is not None:
        warnings.warn(
            "the `scaled` keyword argument is not supported by lalframe, "
            "if you require ADC scaling, please install "
            "python-ldas-tools-framecpp and use `backend='frameCPP'` when "
            "reading GWF data",
            stacklevel=2,
        )

    stream = open_data_source(source)

    # parse times and restrict to available data
    streamstart, streamend = get_stream_segment(stream)
    if start is None:
        startgps = streamstart
    else:
        startgps = lalutils.to_lal_ligotimegps(start)
    if startgps >= streamend:
        msg = f"cannot read data starting at {startgps}, stream ends at {streamend}"
        raise ValueError(msg)
    startgps = max(streamstart, startgps)
    if end is None:
        endgps = streamend
    else:
        endgps = lalutils.to_lal_ligotimegps(end)
    if endgps <= streamstart:
        msg = f"cannot read data ending at {endgps}, stream starts at {streamstart}"
        raise ValueError(msg)
    endgps = min(streamend, endgps)
    span = Segment(startgps, endgps)
    duration = float(abs(span))

    if endgps <= startgps:
        msg = "cannot read data with non-positive duration"
        raise ValueError(msg)

    # read data
    out = series_class.DictClass()
    for name in channels:
        lalframe.FrStreamSeek(stream, startgps)
        # read this channel and convert to GWpy struct
        ts = _read_channel(stream, str(name), start=startgps, duration=duration)
        out[name] = series_class.from_lal(ts, copy=False)
        # check what we got back
        tsend = ts.epoch + getattr(ts.data.data, "size", 0) * ts.deltaT
        if (missing := endgps - tsend) >= ts.deltaT:  # one sample missing
            # if the available data simply didn't go all the way to the end
            # lalframe would have errored (and we protect against that above),
            # so we know there is data missing.
            # see https://git.ligo.org/lscsoft/lalsuite/-/issues/710
            ts = _read_channel(
                stream,
                str(name),
                start=tsend,
                duration=float(missing),
            )
            out[name] = out[name].append(ts.data.data, inplace=False)

    return out


def _read_channel(
    stream: lalframe.FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
) -> lalutils.LALTimeSeriesType:
    """Read a channel from this stream."""
    # find the data type for this channel
    try:
        dtype = lalframe.FrStreamGetTimeSeriesType(channel, stream)
    except RuntimeError as exc:
        if str(exc).lower() == "wrong name":
            msg = f"channel '{channel}' not found"
            raise ValueError(msg) from exc
        raise
    # get the appropriate reader for this type
    reader = lalutils.find_typed_function(
        dtype,
        "FrStreamRead",
        "TimeSeries",
        module=lalframe,
    )
    # read the data
    return reader(stream, channel, start, duration, 0)


# -- write ---------------------------

def write(
    tsdict: TimeSeriesBaseDict,
    outfile: str | Path | IO,
    start: LIGOTimeGPS,
    end: LIGOTimeGPS,
    type: str | None = None,
    name: str | None = None,
    run: int = 0,
    compression: int | str | None = None,
    compression_level: int | None = None,
) -> None:
    """Write data to a GWF file using the LALFrame API."""
    startgps = lalutils.to_lal_ligotimegps(start)
    endgps = lalutils.to_lal_ligotimegps(end)

    ifos = {
        ts.channel.ifo for ts in tsdict.values() if (
            ts.channel
            and ts.channel.ifo
        )
    }

    # LALFrame Python bindings (as we use them) don't allow specifying
    # compression options
    if compression is not None or compression_level is not None:
        warnings.warn(
            "LALFrame backend does not support compression options, "
            "compression and compression_level values will be ignored",
            stacklevel=2,
        )


    # create new frame
    frame = io_gwf_lalframe.create_frame(
        startgps,
        float(endgps - startgps),
        name or "gwpy",
        run,
        0,
        ifos,
    )

    # append each series
    for series in tsdict.values():
        # get type
        ctype = (
            type
            or getattr(series.channel, "_ctype", "proc")
            or "proc"
        ).title()

        # convert to LAL
        lalseries = series.to_lal()

        # find adder
        add_ = lalutils.find_typed_function(
            series.dtype,
            "FrameAdd",
            f"TimeSeries{ctype}Data",
            module=lalframe,
        )

        # add time series to frame
        add_(frame, lalseries)

    # write frame
    io_gwf_lalframe.write_frames(outfile, [frame])
