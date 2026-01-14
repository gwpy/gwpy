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

"""Read data from gravitational-wave frame (GWF) files using |LDAStools.frameCPP|__."""

from __future__ import annotations

import re
from collections import defaultdict
from math import ceil
from typing import TYPE_CHECKING

import numpy

from ....io.gwf import framecpp as io_framecpp
from ....io.utils import file_list
from ....segments import (
    Segment,
    SegmentList,
)
from ....time import (
    LIGOTimeGPS,
    to_gps,
)
from ... import (
    TimeSeries,
    TimeSeriesBaseList,
)
from ...core import _dynamic_scaled
from .utils import _channel_dict_kwarg

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
        Mapping,
    )
    from pathlib import Path
    from typing import (
        IO,
        Concatenate,
        TypeAlias,
        TypeVar,
    )

    import LDAStools.frameCPP

    from ....time import SupportsToGps
    from ... import (
        TimeSeriesBase,
        TimeSeriesBaseDict,
    )

    _TimeSeriesType = TypeVar("_TimeSeriesType", bound=TimeSeriesBase)
    _FrDataType: TypeAlias = (
        LDAStools.frameCPP.FrAdcData
        | LDAStools.frameCPP.FrProcData
        | LDAStools.frameCPP.FrSimData
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# error regexs
FRERR_NO_FRAME_AT_NUM = re.compile(
    r"\ARequest for frame (?P<frnum>\d+) exceeds the range of "
    r"0 through (?P<nframes>\d+)\Z",
)
FRERR_NO_CHANNEL_OF_TYPE = re.compile(
    r"\ANo Fr(Adc|Proc|Sim)Data structures with the name ",
)

FRDATA_TYPES: tuple[str, ...] = (
    "ADC",
    "Proc",
    "Sim",
)


class _Skip(ValueError):  # noqa: N818
    """Error denoting that the contents of a given structure aren't required."""



# -- utilities -----------------------

def get_toc_segments(toc: LDAStools.frameCPP.FrTOC) -> SegmentList:
    """Return the segments for each frame listed in the TOC."""
    out = SegmentList()
    for s, ns, dur in zip(toc.GTimeS, toc.GTimeN, toc.dt, strict=True):
        fstart = LIGOTimeGPS(s, ns)
        out.append(Segment(fstart, fstart + dur))
    return out


def get_frame_segment(frame: LDAStools.frameCPP.FrameH) -> Segment:
    """Return the ``[start, end)`` segment for this frame."""
    epoch = LIGOTimeGPS(*frame.GetGTime())
    return Segment(epoch, epoch + frame.GetDt())


# -- read ----------------------------

def read(
    source: str | Path | IO,
    channels: list[str],
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    scaled: bool | None = None,
    type: str | dict[str, str] | None = None,
    series_class: type[TimeSeriesBase] = TimeSeries,
) -> TimeSeriesBaseDict:
    """Read a dict of series from one or more GWF files.

    Parameters
    ----------
    source : `str`, `pathlib.path`, `file, `list`
        Source of data, any of the following:

        - `str` path of single data file,
        - `str` path of cache file,
        - `list` of paths.

    channels : `~gwpy.detector.ChannelList`, `list`
        A list of channels to read from the source.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str` optional
        GPS start time of required data, anything parseable by
        :func:`~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, anything parseable by
        :func:`~gwpy.time.to_gps` is fine.

    scaled : `bool`, optional
        Apply slope and bias calibration to ADC data.

    type : `dict`, optional
        A `dict` of ``(name, channel-type)`` pairs, where ``channel-type``
        can be one of ``'adc'``, ``'proc'``, or ``'sim'``.

    series_class : `type`, optional
        The `Series` sub-type to return.

    Returns
    -------
    data : `~gwpy.timeseries.TimeSeriesDict` or similar
        A dict of ``(channel, series)`` pairs read from the GWF source(s).
    """
    # parse input source
    files = file_list(source)

    # parse type
    ctype = _channel_dict_kwarg(
        type,
        channels,
        expected_type=str,
        varname="type",
    )

    # read each file individually
    tmp: dict[str, TimeSeriesBaseList] = defaultdict(TimeSeriesBaseList)
    for file_ in files:
        # read each channel and each frame separately
        for series in _read_gwf(
            file_,
            channels,
            start,
            end,
            ctype,
            scaled,
            series_class,
        ):
            if series.name in tmp:
                tmp[series.name].append(series)
            else:
                tmp[series.name].append(numpy.require(series, requirements=["O"]))
    out = series_class.DictClass()
    for channel, serieslist in tmp.items():
        out[channel] = serieslist.join()

    return out


def _read_gwf(
    filename: str,
    channels: list[str],
    start: SupportsToGps | None,
    end: SupportsToGps | None,
    ctype: Mapping[str, str | None],
    scaled: bool | None,
    series_class: type[_TimeSeriesType],
) -> Iterator[_TimeSeriesType]:
    """Read a dict of series data from a single GWF file.

    Parameters
    ----------
    filename : `str`
        the GWF path from which to read

    channels : `~gwpy.detector.ChannelList`, `list`
        a list of channels to read from the source.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str` optional
        GPS start time of required data, anything parseable by
        :func:`~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, anything parseable by
        :func:`~gwpy.time.to_gps` is fine.

    scaled : `bool`, optional
        apply slope and bias calibration to ADC data.

    ctype : `dict`, optional
        a `dict` of ``(name, channel-type)`` pairs, where ``channel-type``
        can be one of ``'adc'``, ``'proc'``, or ``'sim'``.

    series_class : `type`, optional
        the `Series` sub-type to return.

    Returns
    -------
    data : `~gwpy.timeseries.TimeSeriesDict` or similar
        a dict of ``(channel, series)`` pairs read from the GWF file.
    """
    # open file
    stream = io_framecpp.open_gwf(filename, "r")

    # parse start and end of data
    framesegments = get_toc_segments(stream.GetTOC())
    extent = framesegments.extent()
    if start is None:
        start = extent[0]
    else:
        start = to_gps(start)
    if end is None:
        end = extent[1]
    else:
        end = to_gps(end)

    # request segment
    span = Segment(start, end)

    # loop over frames in GWF
    record = set()
    for i, frameseg in enumerate(framesegments):
        # check whether we need this frame at all
        if not frameseg.intersects(span):
            continue

        # get data from each FrVect for each channel
        data = _read_frame(
            stream,
            i,
            frameseg[0],
            channels,
            start,
            end,
            scaled,
            ctype,
            series_class,
        )
        for series in data:
            yield series
            record.add(series.name)

    # if any channels weren't read, something went wrong
    for channel in channels:
        if (name := str(channel)) not in record:
            msg = f"failed to read '{name}' from '{filename}' in interval {span}"
            raise ValueError(msg)


def _read_frame(
    stream: LDAStools.frameCPP.IFrameFStream,
    index: int,
    epoch: LIGOTimeGPS,
    channels: list[str],
    start: LIGOTimeGPS,
    end: LIGOTimeGPS,
    scaled: bool | None,
    ctype: Mapping[str, str | None],
    series_class: type[_TimeSeriesType],
) -> Iterator[_TimeSeriesType]:
    """Read data from a specific frame in this stream."""
    # and read all the channels
    for channel in channels:
        _scaled = _dynamic_scaled(scaled, channel)
        try:
            new = _read_channel(
                stream,
                index,
                channel,
                ctype,
                epoch,
                start,
                end,
                scaled=_scaled,
                series_class=series_class,
            )
        except _Skip:  # don't need this frame for this channel
            continue
        yield from new


def _read_channel(
    stream: LDAStools.frameCPP.IFrameFStream,
    index: int,
    name: str,
    ctype: Mapping[str, str | None],
    epoch: LIGOTimeGPS,
    start: LIGOTimeGPS,
    end: LIGOTimeGPS,
    *,
    scaled: bool = False,
    series_class: type[_TimeSeriesType] = TimeSeries,
) -> Iterator[_TimeSeriesType]:
    """Read a channel from a specific frame in a stream."""
    data = _get_frdata(stream, index, name, ctype=ctype)
    return _read_frdata(
        data,
        epoch,
        start,
        end,
        scaled=scaled,
        series_class=series_class,
    )


def _get_frdata(
    stream: LDAStools.frameCPP.IFrameFStream,
    index: int,
    name: str,
    ctype: Mapping[str, str | None],
) -> _FrDataType:
    """Brute force-ish method to return the FrData structure for a channel.

    This saves on pulling the channel type from the TOC.
    """
    # types to try
    ctypes: tuple[str, ...]
    if ctp := ctype.get(name, None):
        ctypes = (ctp,)
    else:
        ctypes = FRDATA_TYPES

    # try each one
    for ctp in ctypes:
        try:
            _reader = getattr(stream, f"ReadFr{ctp.title()}Data")
        except AttributeError as exc:
            msg = f"invalid channel type '{ctp}' for {name}"
            raise ValueError(msg) from exc
        try:
            return _reader(index, name)
        except IndexError as exc:
            if FRERR_NO_CHANNEL_OF_TYPE.match(str(exc)):
                continue
            raise
    msg = f"channel '{name}' not found"
    raise ValueError(msg)


def _read_frdata(
    frdata: _FrDataType,
    epoch: LIGOTimeGPS,
    start: LIGOTimeGPS,
    end: LIGOTimeGPS,
    *,
    scaled: bool = False,
    series_class: type[_TimeSeriesType] = TimeSeries,
) -> Iterator[_TimeSeriesType]:
    """Read a series from an `FrData` structure.

    Parameters
    ----------
    frdata : `LDAStools.frameCPP.FrAdcData` or similar
        the data structure to read

    epoch : `float`
        the GPS start time of the containing frame
        (`LDAStools.frameCPP.FrameH.GTime`)

    start : `float`
        the GPS start time of the user request

    end : `float`
        the GPS end time of the user request

    scaled : `bool`, optional
        apply slope and bias calibration to ADC data.

    series_class : `type`, optional
        the `Series` sub-type to return.

    Returns
    -------
    series : `~gwpy.timeseries.TimeSeriesBase`
        the formatted data series

    Raises
    ------
    _Skip
        if this data structure doesn't overlap with the requested
        ``[start, end)`` interval.
    """
    datastart = epoch + frdata.GetTimeOffset()

    # check overlap with user-requested span
    if end and datastart >= end:
        raise _Skip

    # get scaling
    try:
        slope = frdata.GetSlope()
        bias = frdata.GetBias()
    except AttributeError:  # not FrAdcData
        slope = None
        bias = None
    else:
        # workaround https://git.ligo.org/ldastools/LDAS_Tools/-/issues/114
        # by forcing the default slope to 1.
        if bias == slope == 0.:
            slope = 1.
        null_scaling = slope == 1. and bias == 0.

    for j in range(frdata.data.size()):
        # we use range(frdata.data.size()) to avoid segfault
        # related to iterating directly over frdata.data
        try:
            new = _read_frvect(
                frdata.data[j],
                datastart,
                start,
                end,
                name=frdata.GetName(),
                series_class=series_class,
            )
        except _Skip:
            continue

        # apply scaling for ADC channels
        if scaled and slope is not None:
            rtype = numpy.result_type(new, slope, bias)
            typechange = not numpy.can_cast(
                rtype,
                new.dtype,
                casting="same_kind",
            )
            # only apply scaling if interesting _or_ if it would lead to a
            # type change, otherwise we are unnecessarily duplicating memory
            if typechange:
                new = new * slope + bias
            elif not null_scaling:
                new *= slope
                new += bias
        elif slope is not None:
            # user has deliberately disabled the ADC calibration, so
            # the stored engineering unit is not valid, revert to 'counts':
            new.override_unit("count")

        yield new


def _read_frvect(
    vect: LDAStools.frameCPP.FrVect,
    epoch: LIGOTimeGPS,
    start: LIGOTimeGPS,
    end: LIGOTimeGPS,
    name: str,
    series_class: type[_TimeSeriesType],
) -> _TimeSeriesType:
    """Read an array from an `FrVect` structure.

    Raises
    ------
    _Skip
        if this vect doesn't overlap with the requested
        ``[start, end)`` interval, or the name doesn't match.
    """
    # only read FrVect with matching name (or no name set)
    #    frame spec allows for arbitrary other FrVects
    #    to hold other information
    if vect.GetName() and name and vect.GetName() != name:
        raise _Skip

    # get array
    arr = vect.GetDataArray()
    nsamp = arr.size

    # and dimensions
    dim = vect.GetDim(0)
    dx = dim.dx
    x0 = dim.startX
    xunit = dim.GetUnitX() or None

    # start and end GPS times of this FrVect
    dimstart = epoch + x0
    dimend = dimstart + nsamp * dx

    # index of first required sample
    nxstart = int(max(0., float(start-dimstart)) / dx)

    # requested start time is after this frame, skip
    if nxstart >= nsamp:
        raise _Skip

    # index of end sample
    if end:
        nxend = int(nsamp - ceil(max(0., float(dimend-end)) / dx))
    else:
        nxend = None

    if nxstart or nxend:
        arr = arr[nxstart:nxend]

    # -- cast as a series

    # get unit
    unit = vect.GetUnitY() or None

    # create array
    series = series_class(
        numpy.require(arr, requirements=["O"]),
        t0=dimstart+nxstart*dx,
        dt=dx,
        name=name,
        channel=name,
        xunit=xunit,
        unit=unit,
        copy=False,
    )

    # add information to channel
    series.channel.sample_rate = series.sample_rate.value
    series.channel.unit = unit
    series.channel.dtype = series.dtype

    return series


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
    """Write data to a GWF file using the frameCPP API."""
    duration = end - start
    ifos = {
        ts.channel.ifo for ts in tsdict.values() if (
            ts.channel
            and ts.channel.ifo
            and ts.channel.ifo in io_framecpp.DetectorLocation.__members__
        )
    }

    # create frame
    frame = io_framecpp.create_frame(
        time=start,
        duration=duration,
        name=name or "gwpy",
        run=run,
        ifos=ifos,
    )

    # append channels
    for i, key in enumerate(tsdict):
        ctype = (
            type
            or getattr(tsdict[key].channel, "_ctype", "proc").lower()
            or "proc"
        )
        if ctype == "adc":
            kw = {"channelid": i}
        else:
            kw = {}
        _append_to_frame(
            frame,
            tsdict[key].crop(start, end),
            ctype=ctype,
            **kw,
        )

    # write frame to file
    io_framecpp.write_frames(
        outfile,
        [frame],
        compression=compression,
        compression_level=compression_level,
    )


def _append_to_frame(
    frame: LDAStools.frameCPP.FrameH,
    timeseries: TimeSeriesBase,
    ctype: str = "proc",
    **kwargs,
) -> None:
    """Append data from a `TimeSeries` to a `~frameCPP.FrameH`.

    Parameters
    ----------
    frame : `~frameCPP.FrameH`
        Frame header to append to.

    timeseries : `TimeSeries`
        The timeseries to append.

    ctype : `str`
        The type of the channel, one of 'adc', 'proc', 'sim'.

    kwargs
        Other keyword arguments are passed to the relevant
        `create_xxx` function.

    See Also
    --------
    gwpy.io.gwf.create_fradcdata
    gwpy.io.gwf.create_frprocdata
    gwpy.io.gwf_create_frsimdata
        For details of the data structure creation, and associated available
        arguments.
    """
    epoch = LIGOTimeGPS(*frame.GetGTime())

    # create the data container
    ctype = ctype.lower()
    create: Callable[Concatenate[TimeSeriesBase, ...], _FrDataType]
    if ctype == "adc":
        create = io_framecpp.create_fradcdata
        append = frame.AppendFrAdcData
    elif ctype == "proc":
        create = io_framecpp.create_frprocdata
        append = frame.AppendFrProcData
    elif ctype == "sim":
        create = io_framecpp.create_frsimdata
        append = frame.AppendFrSimData
    else:
        msg = (
            f"Invalid channel type '{ctype}', please select one of "
            "'adc, 'proc', or 'sim'",
        )
        raise RuntimeError(msg)
    frdata = create(
        timeseries,
        frame_epoch=epoch,
        **kwargs,
    )

    # append an FrVect
    frdata.AppendData(io_framecpp.create_frvect(timeseries))
    append(frdata)
