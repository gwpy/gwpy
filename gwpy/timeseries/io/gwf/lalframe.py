# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

"""Read gravitational-wave frame (GWF) files using the LALFrame API

The frame format is defined in LIGO-T970130 available from dcc.ligo.org.
"""

from __future__ import (absolute_import, division)

import os.path
import warnings

from six import string_types

# import in this order so that lalframe throws the ImportError
# to give the user a bit more information
import lalframe
import lal
from lal.utils import CacheEntry


from ....io.cache import is_cache
from ....io.utils import (FILE_LIKE, file_list)
from ....utils import lal as lalutils
from ... import TimeSeries

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

FRAME_LIBRARY = 'lalframe'


# -- utilities ----------------------------------------------------------------

def open_data_source(source):
    """Open a GWF file source into a `lalframe.XLALFrStream` object

    Parameters
    ----------
    source : `str`, `file`, `list`
        Data source to read.

    Returns
    -------
    stream : `lalframe.FrStream`
        An open `FrStream`.

    Raises
    ------
    ValueError
        If the input format cannot be identified.
    """
    if isinstance(source, FILE_LIKE):
        source = source.name
    if isinstance(source, CacheEntry):
        source = source.path

    # read cache file
    if (isinstance(source, string_types) and
            source.endswith(('.lcf', '.cache'))):
        return lalframe.FrStreamCacheOpen(lal.CacheImport(source))

    # read glue cache object
    if isinstance(source, list) and is_cache(source):
        cache = lal.Cache()
        for entry in file_list(source):
            cache = lal.CacheMerge(cache, lal.CacheGlob(*os.path.split(entry)))
        return lalframe.FrStreamCacheOpen(cache)

    # read lal cache object
    if isinstance(source, lal.Cache):
        return lalframe.FrStreamCacheOpen(source)

    # read single file
    if isinstance(source, string_types):
        return lalframe.FrStreamOpen(*map(str, os.path.split(source)))

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

def read(source, channels, start=None, end=None, series_class=TimeSeries,
         scaled=None):
    """Read data from one or more GWF files using the LALFrame API
    """
    # scaled must be provided to provide a consistent API with frameCPP
    if scaled is not None:
        warnings.warn(
            "the `scaled` keyword argument is not supported by lalframe, "
            "if you require ADC scaling, please install "
            "python-ldas-tools-framecpp",
        )

    stream = open_data_source(source)

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
        end = min(epoch + streamdur, lalutils.to_lal_ligotimegps(end))
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
    dtype = lalframe.FrStreamGetTimeSeriesType(channel, stream)
    reader = lalutils.find_typed_function(dtype, 'FrStreamRead', 'TimeSeries',
                                          module=lalframe)
    return reader(stream, channel, start, duration, 0)


# -- write --------------------------------------------------------------------

def write(tsdict, outfile, start=None, end=None,
          name='gwpy', run=0):
    """Write data to a GWF file using the LALFrame API
    """
    if not start:
        start = list(tsdict.values())[0].xspan[0]
    if not end:
        end = list(tsdict.values())[0].xspan[1]
    duration = end - start

    # get ifos list
    detectors = 0
    for series in tsdict.values():
        try:
            idx = list(lalutils.LAL_DETECTORS.keys()).index(series.channel.ifo)
            detectors |= 1 << 2*idx
        except (KeyError, AttributeError):
            continue

    # create new frame
    frame = lalframe.FrameNew(start, duration, name, run, 0, detectors)

    for series in tsdict.values():
        # convert to LAL
        lalseries = series.to_lal()

        # find adder
        add_ = lalutils.find_typed_function(
            series.dtype, 'FrameAdd', 'TimeSeriesProcData', module=lalframe)

        # add time series to frame
        add_(frame, lalseries)

    # write frame
    lalframe.FrameWrite(frame, outfile)
