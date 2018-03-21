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

from math import ceil

from six import PY2

import numpy

try:
    from LDAStools import frameCPP
except ImportError:
    import frameCPP
    FRAME_LIBRARY = 'frameCPP'
else:
    FRAME_LIBRARY = 'LDAStools.frameCPP'

from ....io import gwf as io_gwf
from ....io.cache import (file_list, file_segment)
from ....time import LIGOTimeGPS
from ... import TimeSeries

from . import channel_dict_kwarg

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# group types that support the buffer interface
if PY2:
    buffer_types = (bytearray, buffer)
else:
    buffer_types = (bytes, bytearray, memoryview)

# get frameCPP type mapping
NUMPY_TYPE_FROM_FRVECT = {
    frameCPP.FrVect.FR_VECT_C: numpy.int8,
    frameCPP.FrVect.FR_VECT_2S: numpy.int16,
    frameCPP.FrVect.FR_VECT_8R: numpy.float64,
    frameCPP.FrVect.FR_VECT_4R: numpy.float32,
    frameCPP.FrVect.FR_VECT_4S: numpy.int32,
    frameCPP.FrVect.FR_VECT_8S: numpy.int64,
    frameCPP.FrVect.FR_VECT_8C: numpy.complex64,
    frameCPP.FrVect.FR_VECT_16C: numpy.complex128,
    frameCPP.FrVect.FR_VECT_STRING: numpy.string_,
    frameCPP.FrVect.FR_VECT_2U: numpy.uint16,
    frameCPP.FrVect.FR_VECT_4U: numpy.uint32,
    frameCPP.FrVect.FR_VECT_8U: numpy.uint64,
}

FRVECT_TYPE_FROM_NUMPY = dict(
    (v, k) for k, v in NUMPY_TYPE_FROM_FRVECT.items())


# -- read ---------------------------------------------------------------------

def read(source, channels, start=None, end=None, type=None,
         series_class=TimeSeries):
    """Read a dict of series from one or more GWF files
    """
    # parse input source
    source = file_list(source)

    # parse type
    ctype = channel_dict_kwarg(type, channels, (str,))

    # read each individually and append
    out = series_class.DictClass()
    for i, file_ in enumerate(source):
        if i == 1:  # force data into fresh memory so that append works
            for name in out:
                out[name] = numpy.require(out[name], requirements=['O'])
        # read frame
        out.append(read_gwf(file_, channels, start=start, end=end,
                            ctype=ctype, series_class=series_class),
                   copy=False)
    return out


def read_gwf(framefile, channels, start=None, end=None, ctype=None,
             series_class=TimeSeries):
    """Read a dict of series data from a single GWF file
    """
    # parse kwargs
    if not start:
        start = 0
    if not end:
        end = 0

    # open file
    stream = frameCPP.IFrameFStream(str(framefile))

    # get number of frames in file
    try:
        nframe = int(stream.GetNumberOfFrames())
    except (AttributeError, ValueError):
        nframe = None

    # if single frame, trust filename to provide GPS epoch of data
    # as required by the file-naming convention
    epochs = None
    try:
        if nframe == 1:
            epochs = [file_segment(framefile)[0]]
    except ValueError:
        pass
    if epochs is None:
        toc = stream.GetTOC()
        epochs = [LIGOTimeGPS(s, n) for s, n in zip(toc.GTimeS, toc.GTimeN)]

    toclist = {}  # only get names once
    for channel in channels:
        # if ctype not declared, find it from the table-of-contents
        if not ctype.get(channel, None):
            toc = stream.GetTOC()
            for typename in ['Sim', 'Proc', 'ADC']:
                if typename not in toclist:
                    get_ = getattr(toc, 'Get%s' % typename)
                    try:
                        toclist[typename] = get_().keys()
                    except AttributeError:
                        toclist[typename] = get_()
                if str(channel) in toclist[typename]:
                    ctype[channel] = typename.lower()
                    break
        # if still not found, channel isn't in the frame
        if not ctype.get(channel, None):
            raise ValueError("Channel %s not found in frame table of contents"
                             % str(channel))

    # find channels
    out = series_class.DictClass()
    for channel in channels:

        name = str(channel)
        read_func = getattr(stream, 'ReadFr%sData' % ctype[channel].title())
        series = None
        i = 0
        while True:
            try:
                data = read_func(i, name)
            except IndexError as exc:
                if 'exceeds the range' in str(exc):  # no more frames
                    break
                else:  # some other problem (likely channel not present)
                    raise
            offset = data.GetTimeOffset()
            datastart = epochs[i] + offset
            i += 1  # increment frame index before any 'continue'
            # check overlap with user-requested span
            if end and datastart >= end and nframe == 1:
                raise ValueError("Cannot read %s from FrVect in %s "
                                 "ending at %s" % (name, framefile, end))
            elif end and datastart >= end:  # don't need this frame
                continue
            try:
                dataend = datastart + data.GetTRange()
            except AttributeError:  # not proc channel
                pass
            else:
                if datastart == dataend:  # tRange not set
                    # tRange is not required, so if it is 0, it may have been
                    # omitted, rather than actually representing an empty
                    # data set
                    pass
                elif start and dataend < start:  # don't need this frame
                    continue
            for j in range(data.data.size()):
                # we use range(data.data.size()) to avoid segfault
                # related to iterating directly over data.data
                vect = data.data[j]

                # only read FrVect with matching name (or no name set)
                #    frame spec allows for arbitrary other FrVects
                #    to hold other information
                if vect.GetName() and vect.GetName() != name:
                    continue
                # decompress data
                arr = vect.GetDataArray()
                dim = vect.GetDim(0)
                dx = dim.dx
                x0 = dim.startX
                if isinstance(arr, buffer_types):
                    arr = numpy.frombuffer(
                        arr, dtype=NUMPY_TYPE_FROM_FRVECT[vect.GetType()])
                # crop to required subset
                dimstart = datastart + x0
                dimend = dimstart + arr.size * dx
                a = int(max(0., float(start-dimstart)) / dx)
                if end:
                    b = int(arr.size - ceil(max(0., float(dimend-end)) / dx))
                else:
                    b = None
                # if file only has ony frame, error on overlap problems
                if a >= arr.size and nframe == 1:  # start too large
                    raise ValueError("Cannot read %s from FrVect in %s "
                                     "starting at %s"
                                     % (name, framefile, start))
                # otherwise just skip to the next frame
                if a >= arr.size:  # skip frame
                    continue
                if a or b:
                    arr = arr[a:b]
                # cast as series or append
                if series is None:
                    # get unit
                    unit = vect.GetUnitY() or None
                    # create array - need require() to prevent segfault
                    series = numpy.require(
                        series_class(arr, t0=dimstart+a*dx, dt=dx, name=name,
                                     channel=name, unit=unit, copy=False),
                        requirements=['O'])
                    # add information to channel
                    series.channel.sample_rate = series.sample_rate.value
                    series.channel.unit = unit
                    series.channel.dtype = series.dtype
                else:
                    series.append(arr)
        if series is None:
            raise ValueError("Failed to read '%s' from file '%s'"
                             % (str(channel), framefile))
        else:
            out[channel] = series

    return out


# -- write --------------------------------------------------------------------

def write(tsdict, outfile, start=None, end=None, name='gwpy', run=0,
          compression=257, compression_level=6):
    """Write data to a GWF file using the frameCPP API
    """
    # set frame header metadata
    if not start:
        starts = set([LIGOTimeGPS(tsdict[key].x0.value) for key in tsdict])
        if len(starts) != 1:
            raise RuntimeError("Cannot write multiple TimeSeries to a single "
                               "frame with different start times, "
                               "please write into different frames")
        start = list(starts)[0]
    if not end:
        ends = set([tsdict[key].span[1] for key in tsdict])
        if len(ends) != 1:
            raise RuntimeError("Cannot write multiple TimeSeries to a single "
                               "frame with different end times, "
                               "please write into different frames")
        end = list(ends)[0]
    duration = end - start
    start = LIGOTimeGPS(start)
    ifos = set([ts.channel.ifo for ts in tsdict.values() if
                ts.channel and ts.channel.ifo and
                hasattr(frameCPP, 'DETECTOR_LOCATION_%s' % ts.channel.ifo)])

    # create frame
    frame = io_gwf.create_frame(time=start, duration=duration, name=name,
                                run=run, ifos=ifos)

    # append channels
    for i, c in enumerate(tsdict):
        try:
            ctype = tsdict[c].channel._ctype or 'proc'
        except AttributeError:
            ctype = 'proc'
        append_to_frame(frame, tsdict[c].crop(start, end),
                        type=ctype, channelid=i)

    # write frame to file
    io_gwf.write_frames(outfile, [frame], compression=compression,
                        compression_level=compression_level)


def append_to_frame(frame, timeseries, type='proc', channelid=0):
    """Append data from a `TimeSeries` to a `~frameCPP.FrameH`

    Parameters
    ----------
    frame : `~frameCPP.FrameH`
        frame object to append to

    timeseries : `TimeSeries`
        the timeseries to append

    type : `str`
        the type of the channel, one of 'adc', 'proc', 'sim'

    channelid : `int`, optional
        the ID of the channel within the group (only used for ADC channels)
    """
    if timeseries.channel:
        channel = str(timeseries.channel)
    else:
        channel = str(timeseries.name)

    offset = timeseries.t0.value - float(LIGOTimeGPS(*frame.GetGTime()))

    # create the data container
    if type.lower() == 'adc':
        frdata = frameCPP.FrAdcData(
            channel,
            0,  # channel group
            channelid,  # channel number in group
            16,  # number of bits in ADC
            timeseries.sample_rate.value,  # sample rate
        )
        append = frame.AppendFrAdcData
    elif type.lower() == 'proc':
        frdata = frameCPP.FrProcData(
            channel,  # channel name
            str(timeseries.name),  # comment
            frameCPP.FrProcData.TIME_SERIES,  # ID as time-series
            frameCPP.FrProcData.UNKNOWN_SUB_TYPE,  # empty sub-type (fseries)
            offset,  # offset of first sample relative to frame start
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
            offset,  # time offset of first sample
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
        str(timeseries.dx.unit), 0)
    # create FrVect
    vect = frameCPP.FrVect(
        timeseries.name or '', FRVECT_TYPE_FROM_NUMPY[timeseries.dtype.type],
        1, dims, str(timeseries.unit))
    # populate FrVect and return
    vect.GetDataArray()[:] = numpy.require(timeseries.value,
                                           requirements=['C'])
    return vect
