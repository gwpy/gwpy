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

"""Read data from gravitational-wave frame (GWF) files using
|LDAStools.frameCPP|__.
"""

import re
from math import ceil

import numpy

from LDAStools import frameCPP  # noqa: F401

from ....io import gwf as io_gwf
from ....io import _framecpp as io_framecpp
from ....io.utils import file_list
from ....segments import Segment
from ....time import (LIGOTimeGPS, to_gps)
from ... import TimeSeries
from ...core import _dynamic_scaled

from . import channel_dict_kwarg

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

FRAME_LIBRARY = 'LDAStools.frameCPP'

# error regexs
FRERR_NO_FRAME_AT_NUM = re.compile(
    r'\ARequest for frame (?P<frnum>\d+) exceeds the range of '
    r'0 through (?P<nframes>\d+)\Z',
)
FRERR_NO_CHANNEL_OF_TYPE = re.compile(
    r'\ANo Fr(Adc|Proc|Sim)Data structures with the name ',
)


class _Skip(ValueError):
    """Error denoting that the contents of a given structure aren't required
    """
    pass


# -- read ---------------------------------------------------------------------

def read(source, channels, start=None, end=None, scaled=None, type=None,
         series_class=TimeSeries):
    # pylint: disable=redefined-builtin
    """Read a dict of series from one or more GWF files

    Parameters
    ----------
    source : `str`, `list`
        Source of data, any of the following:

        - `str` path of single data file,
        - `str` path of cache file,
        - `list` of paths.

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

    type : `dict`, optional
        a `dict` of ``(name, channel-type)`` pairs, where ``channel-type``
        can be one of ``'adc'``, ``'proc'``, or ``'sim'``.

    series_class : `type`, optional
        the `Series` sub-type to return.

    Returns
    -------
    data : `~gwpy.timeseries.TimeSeriesDict` or similar
        a dict of ``(channel, series)`` pairs read from the GWF source(s).
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
        out.append(read_gwf(file_, channels, start=start, end=end, ctype=ctype,
                            scaled=scaled, series_class=series_class),
                   copy=False)
    return out


def read_gwf(filename, channels, start=None, end=None, scaled=None,
             ctype=None, series_class=TimeSeries):
    """Read a dict of series data from a single GWF file

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

    type : `dict`, optional
        a `dict` of ``(name, channel-type)`` pairs, where ``channel-type``
        can be one of ``'adc'``, ``'proc'``, or ``'sim'``.

    series_class : `type`, optional
        the `Series` sub-type to return.

    Returns
    -------
    data : `~gwpy.timeseries.TimeSeriesDict` or similar
        a dict of ``(channel, series)`` pairs read from the GWF file.
    """
    # parse kwargs
    if not start:
        start = 0
    if not end:
        end = 0
    span = Segment(start, end)

    # open file
    stream = io_gwf.open_gwf(filename, 'r')
    nframes = stream.GetNumberOfFrames()

    # find channels
    out = series_class.DictClass()

    # loop over frames in GWF
    i = 0
    while True:
        this = i
        i += 1

        # read frame
        try:
            frame = stream.ReadFrameNSubset(this, 0)
        except IndexError:
            if this >= nframes:
                break
            raise

        # check whether we need this frame at all
        if not _need_frame(frame, start, end):
            continue

        # get epoch for this frame
        epoch = LIGOTimeGPS(*frame.GetGTime())

        # and read all the channels
        for channel in channels:
            _scaled = _dynamic_scaled(scaled, channel)
            try:
                new = _read_channel(stream, this, str(channel),
                                    ctype.get(channel, None),
                                    epoch, start, end, scaled=_scaled,
                                    series_class=series_class)
            except _Skip:  # don't need this frame for this channel
                continue
            try:
                out[channel].append(new)
            except KeyError:
                out[channel] = numpy.require(new, requirements=['O'])

        # if we have all of the data we want, stop now
        if all(span in out[channel].span for channel in out):
            break

    # if any channels weren't read, something went wrong
    for channel in channels:
        if channel not in out:
            msg = "Failed to read {0!r} from {1!r}".format(
                str(channel), filename)
            if start or end:
                msg += ' for {0}'.format(span)
            raise ValueError(msg)

    return out


def _read_channel(stream, num, name, ctype, epoch, start, end,
                  scaled=True, series_class=TimeSeries):
    """Read a channel from a specific frame in a stream
    """
    data = _get_frdata(stream, num, name, ctype=ctype)
    return read_frdata(data, epoch, start, end,
                       scaled=scaled, series_class=series_class)


def _get_frdata(stream, num, name, ctype=None):
    """Brute force-ish method to return the FrData structure for a channel

    This saves on pulling the channel type from the TOC
    """
    ctypes = (ctype,) if ctype else ('adc', 'proc', 'sim')
    for ctype in ctypes:
        _reader = getattr(stream, 'ReadFr{0}Data'.format(ctype.title()))
        try:
            return _reader(num, name)
        except IndexError as exc:
            if FRERR_NO_CHANNEL_OF_TYPE.match(str(exc)):
                continue
            raise
    raise ValueError("no Fr{{Adc,Proc,Sim}}Data structures with the "
                     "name {0}".format(name))


def _need_frame(frame, start, end):
    frstart = LIGOTimeGPS(*frame.GetGTime())
    if end and frstart >= end:
        return False

    frend = frstart + frame.GetDt()
    if start and frend <= start:
        return False

    return True


def read_frdata(frdata, epoch, start, end, scaled=True,
                series_class=TimeSeries):
    """Read a series from an `FrData` structure

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
        raise _Skip()

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

    out = None
    for j in range(frdata.data.size()):
        # we use range(frdata.data.size()) to avoid segfault
        # related to iterating directly over frdata.data
        try:
            new = read_frvect(frdata.data[j], datastart, start, end,
                              name=frdata.GetName(),
                              series_class=series_class)
        except _Skip:
            continue

        # apply scaling for ADC channels
        if scaled and slope is not None:
            rtype = numpy.result_type(new, slope, bias)
            typechange = not numpy.can_cast(
                rtype,
                new.dtype,
                casting='same_kind',
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
            new.override_unit('count')

        if out is None:
            out = new
        else:
            out.append(new)
    return out


def read_frvect(vect, epoch, start, end, name=None, series_class=TimeSeries):
    """Read an array from an `FrVect` structure

    Parameters
    ----------
    vect : `LDASTools.frameCPP.FrVect`
        the frame vector structur to read

    start : `float`
        the GPS start time of the request

    end : `float`
        the GPS end time of the request

    epoch : `float`
        the GPS start time of the containing `FrData` structure

    name : `str`, optional
        the name of the output `series_class`; this is also used
        to ignore ``FrVect`` structures containing other information

    series_class : `type`, optional
        the `Series` sub-type to return.

    Returns
    -------
    series : `~gwpy.timeseries.TimeSeriesBase`
        the formatted data series

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
        raise _Skip()

    # get array
    arr = vect.GetDataArray()
    nsamp = arr.size

    # and dimensions
    dim = vect.GetDim(0)
    dx = dim.dx
    x0 = dim.startX

    # start and end GPS times of this FrVect
    dimstart = epoch + x0
    dimend = dimstart + nsamp * dx

    # index of first required sample
    nxstart = int(max(0., float(start-dimstart)) / dx)

    # requested start time is after this frame, skip
    if nxstart >= nsamp:
        raise _Skip()

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
    series = series_class(arr, t0=dimstart+nxstart*dx, dt=dx, name=name,
                          channel=name, unit=unit, copy=False)

    # add information to channel
    series.channel.sample_rate = series.sample_rate.value
    series.channel.unit = unit
    series.channel.dtype = series.dtype

    return series


# -- write --------------------------------------------------------------------

def write(tsdict, outfile,
          start=None, end=None,
          type=None,
          name='gwpy', run=0,
          compression='GZIP', compression_level=None):
    """Write data to a GWF file using the frameCPP API

    Parameters
    ----------
    tsdict : `TimeSeriesDict`
        dict of data to write

    outfile : `str`
        the file name of the target output file

    start : `float`, optional
        the GPS start time of the file

    end : `float`, optional
        the GPS end time of the file

    type : `str`, optional
        the type of the channel, one of 'adc', 'proc', 'sim', default
        is 'proc' unless stored in the channel structure

    name : `str`, optional
        the name of each frame

    run : `int`, optional
        the FrameH run number

    compression : `int`, `str`, optional
        name of compresion algorithm to use, or its endian-appropriate
        ID, choose from

        - ``'RAW'``
        - ``'GZIP'``
        - ``'DIFF_GZIP'``
        - ``'ZERO_SUPPRESS_WORD_2'``
        - ``'ZERO_SUPPRESS_WORD_4'``
        - ``'ZERO_SUPPRESS_WORD_8'``
        - ``'ZERO_SUPPRESS_OTHERWISE_GZIP'``

    compression_level : `int`, optional
        compression level for given method, default is ``6`` for GZIP-based
        methods, otherwise ``0``
    """
    # set frame header metadata
    if not start or not end:
        starts, ends = zip(*(ts.span for ts in tsdict.values()))
        start = to_gps(start or min(starts))
        end = to_gps(end or max(ends))
    duration = end - start
    ifos = {
        ts.channel.ifo for ts in tsdict.values() if (
            ts.channel
            and ts.channel.ifo
            and ts.channel.ifo in io_framecpp.DetectorLocation.__members__
        )
    }

    # create frame
    frame = io_gwf.create_frame(
        time=start,
        duration=duration,
        name=name,
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
        if ctype == 'adc':
            kw = {"channelid": i}
        else:
            kw = {}
        _append_to_frame(frame, tsdict[key].crop(start, end), type=ctype, **kw)

    # write frame to file
    io_gwf.write_frames(
        outfile,
        [frame],
        compression=compression,
        compression_level=compression_level,
    )


def _append_to_frame(frame, timeseries, type='proc', **kwargs):
    # pylint: disable=redefined-builtin
    """Append data from a `TimeSeries` to a `~frameCPP.FrameH`

    Parameters
    ----------
    frame : `~frameCPP.FrameH`
        frame object to append to

    timeseries : `TimeSeries`
        the timeseries to append

    type : `str`
        the type of the channel, one of 'adc', 'proc', 'sim'

    **kwargs
        other keyword arguments are passed to the relevant
        `create_xxx` function

    See also
    --------
    gwpy.io.gwf.create_fradcdata
    gwpy.io.gwf.create_frprocdata
    gwpy.io.gwf_create_frsimdata
        for details of the data structure creation, and associated available
        arguments
    """
    epoch = LIGOTimeGPS(*frame.GetGTime())

    # create the data container
    if type.lower() == 'adc':
        create = io_gwf.create_fradcdata
        append = frame.AppendFrAdcData
    elif type.lower() == 'proc':
        create = io_gwf.create_frprocdata
        append = frame.AppendFrProcData
    elif type.lower() == 'sim':
        create = io_gwf.create_frsimdata
        append = frame.AppendFrSimData
    else:
        raise RuntimeError("Invalid channel type {!r}, please select one of "
                           "'adc, 'proc', or 'sim'".format(type))
    frdata = create(timeseries, frame_epoch=epoch, **kwargs)

    # append an FrVect
    frdata.AppendData(io_gwf.create_frvect(timeseries))
    append(frdata)
    return frdata
