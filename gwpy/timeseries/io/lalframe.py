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

"""Read gravitational-wave frame (GWF) files using the LALFrame API.

The frame format is defined in LIGO-T970130 available from dcc.ligo.org.
"""

from __future__ import (division, absolute_import)

import os.path
import tempfile
from six import string_types

import numpy

try:
    from glue.lal import Cache as GlueCache, CacheEntry as GlueCacheEntry
except ImportError:
    GlueCache = GlueCacheEntry = type(None)

from ...io.registry import (register_reader, register_writer,
                            register_identifier)
from ...io.cache import (FILE_LIKE, open_cache, find_contiguous)
from ...io.utils import identify_factory
from ...utils import import_method_dependency
from .. import (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict)
from .cache import read_cache


# -- utilities ----------------------------------------------------------------

def open_data_source(source):
    """Open a GWF file source into a `lalframe.XLALFrStream` object

    Parameters
    ----------
    source : `str`, `file`, `lal.Cache`, `glue.lal.Cache`
        data source to read

    Returns
    -------
    stream : `lalframe.FrStream`
        an open `FrStream`

    Raises
    ------
    ValueError
        if the input format cannot be identified
    """
    lalframe = import_method_dependency('lalframe')
    import lal

    if isinstance(source, (file, tempfile._TemporaryFileWrapper)):  # pylint:disable=protected-access
        source = source.name
    if isinstance(source, GlueCacheEntry):
        source = source.path

    # read single file
    if isinstance(source, string_types) and source.endswith('.gwf'):
        return lalframe.FrStreamOpen(*os.path.split(source))
    # read cache file
    elif (isinstance(source, string_types) and
          source.endswith(('.lcf', '.cache'))):
        return lalframe.FrStreamCacheOpen(lal.CacheImport(source))
    # read glue cache object
    elif isinstance(source, GlueCache):
        cache = lal.Cache()
        for entry in source:
            cache = lal.CacheMerge(
                cache, lal.CacheGlob(*os.path.split(entry.path)))
        return lalframe.FrStreamCacheOpen(cache)
    elif isinstance(source, lal.Cache):
        return lalframe.FrStreamCacheOpen(source)

    raise ValueError("Don't know how to open data source of type %r"
                     % type(source))


def get_stream_duration(stream):
    """Find the duration of time stored in a frame stream

    Parameters
    ----------
    stream : `lal.FrStream`
        stream of data to search

    Returns
    -------
    duration : `float`
        the duration (seconds) of the data for this channel
    """
    lalframe = import_method_dependency('lalframe')
    import lal
    epoch = lal.LIGOTimeGPS(stream.epoch.gpsSeconds,
                            stream.epoch.gpsNanoSeconds)
    # loop over each file in the stream cache and query its duration
    nfile = stream.cache.length
    duration = 0
    for dummy_i in range(nfile):
        for dummy_j in range(lalframe.FrFileQueryNFrame(stream.file)):
            duration += lalframe.FrFileQueryDt(stream.file, 0)
            lalframe.FrStreamNext(stream)
    # rewind stream and return
    lalframe.FrStreamSeek(stream, epoch)
    return duration


# -- read ---------------------------------------------------------------------

def read_timeseriesdict(source, channels, start=None, end=None, dtype=None,
                        resample=None, gap=None, pad=None, nproc=1,
                        series_class=TimeSeries):
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

    pad : `float`, optional
        value with which to fill gaps in the source data, if not
        given gaps will result in an exception being raised

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
    if nproc > 1:
        return read_cache(source, channels, start=start, end=end,
                          gap=gap, pad=pad, resample=resample, dtype=dtype,
                          nproc=nproc, format='gwf',
                          target=series_class.DictClass)

    # parse output format kwargs
    if not isinstance(resample, dict):
        resample = dict((c, resample) for c in channels)
    if not isinstance(dtype, dict):
        dtype = dict((c, dtype) for c in channels)

    # format gap handling
    if gap is None and pad is not None:
        gap = 'pad'
    elif gap is None:
        gap = 'raise'

    # read cache file up-front
    if (isinstance(source, string_types) and
            source.endswith(('.lcf', '.cache'))) or (
            isinstance(source, FILE_LIKE) and
            source.name.endswith(('.lcf', '.cache'))):
        source = open_cache(source)
    # separate cache into contiguous segments
    if isinstance(source, GlueCache):
        source = list(find_contiguous(source))
    # convert everything else into a list if needed
    if (not isinstance(source, (list, tuple)) or
            isinstance(source, GlueCache)):
        source = [source]

    # now read the data
    out = series_class.DictClass()
    for i, src in enumerate(source):
        if i == 1:  # force data into fresh memory so that append works
            for name in out:
                out[name] = numpy.require(out[name], requirements=['O'])
        stream = open_data_source(src)
        out.append(read_stream(stream, channels, start=start, end=end,
                               series_class=series_class),
                   gap=gap, pad=pad, copy=False)
    for name in out:
        if (resample.get(name) and
                resample[name] != out[name].sample_rate.value):
            out[name] = out[name].resample(resample[name])
        if dtype.get(name) and numpy.dtype(dtype[name]) != out[name].dtype:
            out[name] = out[name].astype(dtype[name])
    return out


def read_stream(stream, channels, start=None, end=None,
                series_class=TimeSeries):
    """Read a `TimeSeriesDict` of data from a `FrStream`

    Parameters
    ----------
    stream : `lalframe.FrStream`
        GWF data sream to read

    channels : `list` of `str`
        list of channel names to read

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    Returns
    -------
    timeseriesdict : `TimeSeriesDict`
        a dict of `(channel, TimeSeries)` pairs of data read from the stream
    """
    lalframe = import_method_dependency('lalframe')
    import lal
    from gwpy.utils import lal as lalutils

    # parse times and restrict to available data
    epoch = lal.LIGOTimeGPS(stream.epoch.gpsSeconds,
                            stream.epoch.gpsNanoSeconds)
    streamdur = get_stream_duration(stream)
    if start is None:
        start = epoch
    else:
        start = max(epoch, lalutils.to_lal_ligotimegps(start))
    if end is None:
        offset = float(start - epoch)
        duration = streamdur - offset
    else:
        end = min(epoch + streamdur, end)
        duration = float(end - start)

    # read data
    out = series_class.DictClass()
    for name in channels:
        out[name] = series_class.from_lal(
            _read_channel(stream, str(name), start=start, duration=duration),
            copy=False)
        lalframe.FrStreamSeek(stream, epoch)
    return out


def _read_channel(stream, channel, start, duration):
    lalframe = import_method_dependency('lalframe')
    from gwpy.utils import lal as lalutils
    dtype = lalframe.FrStreamGetTimeSeriesType(channel, stream)
    typestr = lalutils.LAL_TYPE_STR[dtype]
    reader = getattr(lalframe, 'FrStreamRead%sTimeSeries' % typestr)
    return reader(stream, channel, start, duration, 0)


def read_timeseries(source, channel, *args, **kwargs):
    """Read `TimeSeries` from GWF source
    """
    return read_timeseriesdict(source, [channel], *args, **kwargs)[channel]


def read_statevectordict(source, channels, *args, **kwargs):
    """Read `StateVectorDict` from GWF source
    """
    bitss = kwargs.pop('bits', {})
    # read list of bit lists
    if (isinstance(bitss, (list, tuple)) and len(bitss) and
            isinstance(bitss[0], (list, tuple))):
        bitss = dict(zip(channels, bitss))
    # read single list for all channels
    elif isinstance(bitss, (list, tuple)):
        bitss = dict((channel, bitss) for channel in channels)
    # otherwise assume dict of bit lists

    # read data as timeseriesdict and repackage with bits
    kwargs.setdefault('series_class', StateVector)
    svd = StateVectorDict(
        read_timeseriesdict(source, channels, *args, **kwargs))
    for (channel, bits) in bitss.iteritems():
        svd[channel].bits = bits
    return svd


def read_statevector(source, channel, *args, **kwargs):
    """Read `StateVector` from GWF source
    """
    bits = kwargs.pop('bits', None)
    kwargs.setdefault('series_class', StateVector)
    statevector = read_timeseries(source, channel, *args, **kwargs)
    statevector.bits = bits
    return statevector


# -- write --------------------------------------------------------------------

def write_timeseriesdict(tsdict, outfile, start=None, end=None,
                         project='gwpy', run=0, frame=0):
    """Write a `TimeSeriesDict` to disk in GWF format

    Parameters
    ----------
    tsdict : `TimeSeriesDict`
        the data to write

    outfile : `str`
        the path of the output frame file

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    project : `str`, optional
        name to write into frame header

    run : `int`, optional
        run number to write into frame header

    frame : `int`, optional
        frame number to write into frame header
    """
    lalframe = import_method_dependency('lalframe')
    import lal
    from gwpy.utils import lal as lalutils

    if not start:
        start = tsdict.values()[0].xspan[0]
    if not end:
        end = tsdict.values()[0].xspan[1]
    duration = end - start

    # get ifos list
    try:
        detectors = 0
        for series in tsdict.values():
            idx = list(lalutils.LAL_DETECTORS.keys()).index(series.channel.ifo)
            detectors |= 2**idx
    except (KeyError, AttributeError):
        detectors = lal.LALDETECTORTYPE_ABSENT

    # create new frame
    frame = lalframe.FrameNew(start, duration, project, run, frame, detectors)

    for series in tsdict.values():
        # convert to LAL
        lalseries = series.to_lal()

        # find adder
        laltype = lalutils.LAL_TYPE_FROM_NUMPY[series.dtype.type]
        typestr = lalutils.LAL_TYPE_STR[laltype]
        add_ = getattr(lalframe, 'FrameAdd%sTimeSeriesProcData' % typestr)

        # add time series to frame
        add_(frame, lalseries)

    # write frame
    lalframe.FrameWrite(frame, outfile)


def write_timeseries(timeseries, outfile, *args, **kwargs):
    """Write a `TimeSeries` to disk in GWF file format

    Parameters
    ----------
    timeseries : `TimeSeries`
        the data series to write

    outfile : `str`
        the path of the output frame file

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    project : `str`, optional
        name to write into frame header

    run : `int`, optional
        run number to write into frame header

    frame : `int`, optional
        frame number to write into frame header
    """
    return write_timeseriesdict(
        {timeseries.name: timeseries}, outfile, *args, **kwargs)


# -- registration -------------------------------------------------------------

# TODO: remove force=True when frameCPP reading is removed

register_reader('gwf', TimeSeries, read_timeseries, force=True)
register_reader('gwf', TimeSeriesDict, read_timeseriesdict, force=True)
register_reader('gwf', StateVector, read_statevector, force=True)
register_reader('gwf', StateVectorDict, read_statevectordict, force=True)
register_writer('gwf', TimeSeries, write_timeseries, force=True)
register_writer('gwf', TimeSeriesDict, write_timeseriesdict, force=True)
register_writer('gwf', StateVector, write_timeseries, force=True)
register_writer('gwf', StateVectorDict, write_timeseriesdict, force=True)
_identify = identify_factory('gwf')  # pylint:disable=invalid-name
for cls in (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict):
    register_identifier('gwf', cls, _identify, force=True)
