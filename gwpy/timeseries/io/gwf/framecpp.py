# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Read data from gravitational-wave frames using frameCPP.
"""

from __future__ import division
import __builtin__

import numpy

from ....io.cache import (CacheEntry, file_list)
from ....time import LIGOTimeGPS
from ....segments import Segment
from ....utils import (gprint, with_import)
from ... import (TimeSeries, TimeSeriesDict)

from . import channel_dict_kwarg

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# get frameCPP path
try:
    from LDAStools.frameCPP import FrVect
except ImportError:
    DEPENDS = 'frameCPP'
    try:
        from frameCPP import FrVect
    except ImportError:
        FrVect = None
else:
    DEPENDS = 'LDAStools.frameCPP'

# get frameCPP type mapping
if FrVect is not None:
    NUMPY_TYPE_FROM_FRVECT = {
        FrVect.FR_VECT_C: numpy.int8,
        FrVect.FR_VECT_2S: numpy.int16,
        FrVect.FR_VECT_8R: numpy.float64,
        FrVect.FR_VECT_4R: numpy.float32,
        FrVect.FR_VECT_4S: numpy.int32,
        FrVect.FR_VECT_8S: numpy.int64,
        FrVect.FR_VECT_8C: numpy.complex64,
        FrVect.FR_VECT_16C: numpy.complex128,
        FrVect.FR_VECT_STRING: numpy.string_,
        FrVect.FR_VECT_2U: numpy.uint16,
        FrVect.FR_VECT_4U: numpy.uint32,
        FrVect.FR_VECT_8U: numpy.uint64,
    }
# default to LDASTools v2.2.1
else:
    NUMPY_TYPE_FROM_FRVECT = {
        0: numpy.int8,
        1: numpy.int16,
        2: numpy.float64,
        3: numpy.float32,
        4: numpy.int32,
        5: numpy.int64,
        6: numpy.complex64,
        7: numpy.complex128,
        8: numpy.string_,
        9: numpy.uint16,
        10: numpy.uint32,
        11: numpy.uint64,
        12: numpy.uint8,
    }

FRVECT_TYPE_FROM_NUMPY = dict(
    (v, k) for k, v in NUMPY_TYPE_FROM_FRVECT.items())


@with_import(DEPENDS)
def read_timeseriesdict(source, channels, start=None, end=None, type=None,
                        dtype=None, resample=None, verbose=False,
                        _SeriesClass=TimeSeries):
    """Read the data for a list of channels from a GWF data source.

    Parameters
    ----------
    source : `str`, :class:`glue.lal.Cache`, `list`
        data source object, one of:

        - `str` : frame file path
        - :class:`glue.lal.Cache`, `list` : contiguous list of frame paths

    channels : `list`
        list of channel names (or `Channel` objects) to read from frame.
    start : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        start GPS time of desired data.
    end : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        end GPS time of desired data.
    type : `str`
        type of channel, one of ``'adc'`` or ``'proc'``.
    dtype : `numpy.dtype`, `str`, `type`, or `dict`
        numeric data type for returned data, e.g. `numpy.float`, or
        `dict` of (`channel`, `dtype`) pairs
    resample : `float`, optional
        rate of samples per second at which to resample input TimeSeries.
    verbose : `bool`, optional
        print verbose output.

    Returns
    -------
    dict : :class:`~gwpy.timeseries.TimeSeriesDict`
        dict of (channel, `TimeSeries`) data pairs

    Notes
    -----
    If reading from a list, or cache, or framefiles, the frames contained
    must be contiguous and sorted in chronological order for this function
    to return without exception.

    Raises
    ------
    ValueError
        if reading from an unsorted, or discontiguous cache of files
    """
    frametype = None
    # parse input source
    filelist = file_list(source)
    try:
        frametype = CacheEntry.from_T050017(filelist[0]).description
    except ValueError:
        frametype = None
    # parse resampling
    resample = channel_dict_kwarg(resample, channels, (int,))
    if resample is None:
        raise ValueError("Cannot parse `resample` request, please review "
                         "documentation for that argument")
    # parse type
    type = channel_dict_kwarg(type, channels, (str,))
    if resample is None:
        raise ValueError("Cannot parse channel `type` request, please review "
                         "documentation for that argument")
    # parse dtype
    dtype = channel_dict_kwarg(dtype, channels, (str, __builtin__.type),
                               astype=numpy.dtype)
    if dtype is None:
        raise ValueError("Cannot parse `dtype` request, please review "
                         "documentation for that argument")
    # read each individually and append
    N = len(filelist)
    if verbose:
        if not isinstance(verbose, (unicode, str)):
            verbose = ''
        gprint("%sReading %d channels from frames... 0/%d (0.00%%)\r"
               % (verbose, len(channels), N), end='')
    out = TimeSeriesDict()
    for i, fp in enumerate(filelist):
        # read frame
        new = _read_frame(fp, channels, start=start, end=end, ctype=type,
                          dtype=dtype, _SeriesClass=_SeriesClass)
        # get channel type for next frame (means we only query the TOC once)
        if not i:
            for channel, ts in new.iteritems():
                type[channel] = ts.channel._ctype
        # store
        out.append(new, copy=False)
        if verbose is not False:
            gprint("%sReading %d channels from frames... %d/%d (%.1f%%)\r"
                   % (verbose, len(channels), i+1, N, (i+1)/N * 100), end='')
    if verbose is not False:
        gprint("%sReading %d channels from frames... %d/%d (100.0%%)"
               % (verbose, len(channels), N, N))
    # finalise
    for channel, ts in out.iteritems():
        ts.channel.frametype = frametype
        # resample data
        if resample is not None and channel in resample:
            out[channel] = out[channel].resample(resample[channel])
        # crop data
        if start is not None or end is not None:
            out[channel] = out[channel].crop(start=start, end=end)
        # copy into fresh memory if needed
        out[channel] = numpy.require(out[channel], requirements=['O'])
    return out


def _read_frame(framefile, channels, start=None, end=None, ctype=None,
                dtype=None, _SeriesClass=TimeSeries):
    """Internal function to read data from a single frame.

    All users should be using the wrapper `read_timeseriesdict`.

    Parameters
    ----------
    framefile : `str`, :class:`~glue.lal.CacheEntry`
        path to GWF-format frame file on disk.
    channels : `list`
        list of channels to read.
    start : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        start GPS time of desired data.
    end : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        end GPS time of desired data.
    ctype : `str`, optional
        channel data type to read, one of: ``'adc'``, ``'proc'``.
    dtype : `numpy.dtype`, `str`, `type`, `dict`
    _SeriesClass : `type`, optional
        class object to use as the data holder for a single channel,
        default is :class:`~gwpy.timeseries.TimeSeries`

    Returns
    -------
    dict : :class:`~gwpy.timeseries.TimeSeriesDict`
        dict of (channel, `TimeSeries`) data pairs
    """
    if isinstance(channels, (unicode, str)):
        channels = channels.split(',')

    # construct span segment
    span = Segment(start is not None and start or -numpy.inf,
                   end is not None and end or numpy.inf)

    # open file
    if isinstance(framefile, CacheEntry):
        fp = framefile.path
    else:
        fp = framefile
    stream = frameCPP.IFrameFStream(fp)

    # interpolate frame epochs from CacheEntry
    try:
        nframe = int(stream.GetNumberOfFrames())
    except (AttributeError, ValueError):
        nframe = None
    if isinstance(framefile, CacheEntry) and nframe == 1:
        epochs = [float(framefile.segment[0])]
    else:
        epochs = None

    # load table of contents if needed
    if epochs is None or not ctype:
        toc = stream.GetTOC()
    # get list of frame epochs
    if epochs is None:
        epochs = toc.GTimeS
    # work out channel types
    if not ctype:
        try:
            adcs = toc.GetADC().keys()
        except AttributeError:
            adcs = toc.GetADC()
        try:
            procs = toc.GetProc().keys()
        except AttributeError:
            procs = toc.GetProc()
        ctype = {}
        for channel in channels:
            name = str(channel)
            if name in adcs:
                ctype[channel] = 'adc'
            elif name in procs:
                ctype[channel] = 'proc'
            else:
                raise ValueError("Channel %s not found in frame table of "
                                 "contents" % name)

    # set output
    out = TimeSeriesDict()
    for channel in channels:
        name = str(channel)
        read_ = getattr(stream, 'ReadFr%sData' % ctype[channel].title())
        ts = None
        i = 0
        dtype_ = dtype.get(channel, None)
        while True:
            try:
                data = read_(i, name)
            except IndexError as e:
                if 'exceeds the range' in str(e):  # no more frames
                    break
                else:  # some other problem (likely channel not present)
                    raise
            offset = data.GetTimeOffset()
            thisepoch = epochs[i] + offset
            try:
                thisspan = Segment(thisepoch, thisepoch + data.GetTRange())
            except AttributeError:
                pass
            else:
                if not thisspan.intersects(span):
                    i += 1
                    continue
            for vect in data.data:
                arr = vect.GetDataArray()
                if isinstance(arr, buffer):
                    arr = numpy.frombuffer(
                        arr, dtype=NUMPY_TYPE_FROM_FRVECT[vect.GetType()])
                dx = vect.GetDim(0).dx
                if ts is None:
                    # create array
                    unit = vect.GetUnitY() or None
                    ts = numpy.require(
                        _SeriesClass(arr, epoch=thisepoch, dx=dx, name=name,
                                     channel=channel, unit=unit, dtype=dtype_,
                                     copy=False), requirements=['O'])
                    if not ts.channel.dtype:
                        ts.channel.dtype = arr.dtype
                    ts.channel._ctype = ctype[channel]
                elif arr.dtype != ts.dtype:
                    ts.append(arr.astype(dtype_))
                else:
                    ts.append(arr)
            i += 1
        if ts is None:
            raise ValueError("Channel '%s' not found in frame '%s'"
                             % (str(channel), fp))
        else:
            out[channel] = ts

    return out


@with_import(DEPENDS)
def write_timeseriesdict(tsdict, outfile, start=None, end=None):
    """Write a GWF file containing data from a `TimeSeriesDict`

    Parameters
    ----------
    tsdict : `TimeSeriesDict`
        data to write
    outfile : `str`
        the output file path to write to
    start : `~gwpy.time.LIGOTimeGPS`, optional
        the desired GPS start time of the output data
        (defaults to start of given data)
    end : `~gwpy.time.LIGOTimeGPS`, optional
        the desired GPS end time of the output data
        (defaults to end of given data)
    """
    stream = frameCPP.OFrameFStream(outfile)
    frame = create_frame(tsdict, start=start, end=end)
    stream.WriteFrame(
        frame,  # frame to write
        frameCPP.FrVect.RAW,  # compression scheme
        0,  # compession level
    )


@with_import(DEPENDS)
def create_frame(tsdict, start=None, end=None):
    """Create a new `~frameCPP.FrameH` and fill it with data

    Parameters
    ----------
    tsdict : `TimeSeriesDict`
        a `dict` of `(key, timeseries)` pairs
    start : `~gwpy.time.LIGOTimeGPS`, optional
        the desired GPS start time of the output data
        (defaults to start of given data)
    end : `~gwpy.time.LIGOTimeGPS`, optional
        the desired GPS end time of the output data
        (defaults to end of given data)

    Returns
    -------
    frame : `~frameCPP.FrameH`
        a new frame filled with data
    """
    if not start:
        starts = set([LIGOTimeGPS(tsdict[c].x0.value) for c in tsdict])
        durations = set([tsdict[c].duration.value for c in tsdict])
        if len(starts) != 1:
            raise RuntimeError("Cannot write multiple TimeSeries to a single "
                               "frame with different start times, "
                               "please write into different frames")
        start = list(starts)[0]
    if not end:
        ends = set([tsdict[c].span[1] for ts in tsdict])
        if len(ends) != 1:
            raise RuntimeError("Cannot write multiple TimeSeries to a single "
                               "frame with different end times, "
                               "please write into different frames")
        end = list(ends)[0]
    duration = end - start
    # create frame
    frame = frameCPP.FrameH()
    # set start time
    start = LIGOTimeGPS(start)
    frame.SetGTime(frameCPP.GPSTime(start.seconds, start.nanoseconds))
    # set duration
    frame.SetDt(float(duration))
    # append channels
    for i, c in enumerate(tsdict):
        try:
            ctype = tsdict[c].channel._ctype or 'proc'
        except AttributeError:
            ctype = 'proc'
        append_to_frame(frame, tsdict[c].crop(start, end),
                        type=ctype, channelid=i)
    return frame


@with_import(DEPENDS)
def append_to_frame(frame, timeseries, type='proc',
                    channelid=0, channelgroup=0):
    """Append data from a `TimeSeries` to a `~frameCPP.FrameH`

    Parameters
    ----------
    frame : `~frameCPP.FrameH`
        frame object to append to
    timeseries : `TimeSeries`
        the timeseries to append
    type : `str`
        the type of the channel, one of 'adc', 'proc', 'sim'
    channelgroup : `int`, optional
        the ID of the group for this channel
    channelid : `int`, optional
        the ID of the channel within the group
    """
    # create the data container
    if type.lower() == 'adc':
        frdata = frameCPP.FrAdcData(
            str(timeseries.channel),
            0,  # channel group
            channelid,  # channel number in group
            16,  # number of bits in ADC
            timeseries.sample_rate.value,  # sample rate
        )
        append = frame.AppendFrAdcData
    elif type.lower() == 'proc':
        frdata = frameCPP.FrProcData(
            str(timeseries.channel),  # channel name
            str(timeseries.name),  # comment
            frameCPP.FrProcData.TIME_SERIES,  # ID as time-series
            frameCPP.FrProcData.UNKNOWN_SUB_TYPE,  # empty sub-type (fseries)
            0.,  # offset of first sample relative to frame start
            abs(timeseries.span),  # duration of data
            0.,  # heterodyne frequency
            0.,  # phase of heterodyne
            0.,  # frequency range
            0.,  # resolution bandwidth
        )
        append = frame.AppendFrProcData
    elif type.lower() == 'sim':
        frdata = frameCPP.FrSimData(
            str(timeseries.channel),  # channel name
            str(timeseries.name),  # comment
            timeseries.sample_rate.value,  # sample rate
            0.,  # time offset of first sample
            0.,  # heterodyne frequency
            0.,  # phase of heterodyne
        )
        append = frame.AppendFrSimData
    else:
        raise RuntimeError("Invalid channel type %r, please select one of "
                           "'adc, 'proc', or 'sim'" % type)
    # append an FrVect
    frdata.AppendData(create_frvect(timeseries))
    append(frdata)


@with_import(DEPENDS)
def create_frvect(timeseries):
    """Create a `~frameCPP.FrVect` from a `TimeSeries`

    This method is primarily designed to make writing data to GWF files a
    bit easier.

    Parameters
    ----------
    timeseries : `TimeSeries`
        the input `TimeSeries`

    Returns
    -------
    frvect : `~frameCPP.FrVect`
        the output `FrVect`
    """
    # create timing dimension
    dims = frameCPP.Dimension(
        timeseries.size, timeseries.dx.value,
        str(timeseries.dx.unit), timeseries.x0.value)
    # create FrVect
    vect = frameCPP.FrVect(
        timeseries.name or '', FRVECT_TYPE_FROM_NUMPY[timeseries.dtype.type],
        1, dims, str(timeseries.unit))
    # populate FrVect and return
    vect.GetDataArray()[:] = timeseries.value
    return vect
