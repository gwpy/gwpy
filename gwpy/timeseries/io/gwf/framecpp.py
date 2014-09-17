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

Direct access to the frameCPP library is the easiest way to read multiple
channels from a single frame file in one go.
"""

from __future__ import division

from astropy.io import registry

from glue.lal import (Cache, CacheEntry)

from .identify import register_identifier
from .... import version
from ....utils import (gprint, with_import)
from ... import (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


@with_import('frameCPP')
def read_timeseriesdict(source, channels, start=None, end=None, type=None,
                        resample=None, verbose=False, _SeriesClass=TimeSeries):
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
    resample : `float`, optional
        rate of samples per second at which to resample input TimeSeries.
    verbose : `bool`, optional
        print verbose output.

    Returns
    -------
    dict : :class:`~gwpy.timeseries.core.TimeSeriesDict`
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
    # parse input source
    if isinstance(source, file):
        filelist = [source.name]
    elif isinstance(source, (unicode, str)):
        filelist = source.split(',')
    elif isinstance(source, CacheEntry):
        filelist = [source]
    elif isinstance(source, Cache):
        source.sort(key=lambda e: e.segment[0])
        filelist = source
    else:
        filelist = list(source)
    # parse resampling
    if isinstance(resample, int):
        resample = dict((channel, resample) for channel in channels)
    elif isinstance(resample, (tuple, list)):
        resample = dict(zip(channels, resample))
    elif resample is None:
        resample = {}
    elif not isinstance(resample, dict):
        raise ValueError("Cannot parse resample request, please review "
                         "documentation for that argument")
    # read each individually and append
    N = len(filelist)
    if verbose:
        if not isinstance(verbose, (unicode, str)):
            verbose = ''
        gprint("%sReading %d channels from frames... 0/%d (0.00%%)\r"
               % (verbose, len(channels), N), end='')
    out = TimeSeriesDict()
    for i,fp in enumerate(filelist):
        # read frame
        new = _read_frame(fp, channels, type=type, verbose=verbose,
                          _SeriesClass=_SeriesClass)
        # store
        out.append(new)
        if verbose is not False:
            gprint("%sReading %d channels from frames... %d/%d (%.1f%%)\r"
                   % (verbose, len(channels), i+1, N, (i+1)/N * 100), end='')
    if verbose is not False:
        gprint("%sReading %d channels from frames... %d/%d (100.0%%)"
               % (verbose, len(channels), N, N))
    # finalise
    for channel, ts in out.iteritems():
        # resample data
        if resample is not None and channel in resample:
            out[channel] = out[channel].resample(resample[channel])
        # crop data
        if start is not None or end is not None:
            out[channel] = out[channel].crop(start=start, end=end)
    return out


def _read_frame(framefile, channels, type=None, verbose=False,
                _SeriesClass=TimeSeries):
    """Internal function to read data from a single frame.

    All users should be using the wrapper `read_timeseriesdict`.

    Parameters
    ----------
    framefile : `str`, :class:`~glue.lal.CacheEntry`
        path to GWF-format frame file on disk.
    channels : `list`
        list of channels to read.
    type : `str`, optional
        channel data type to read, one of: ``'adc'``, ``'proc'``.
    verbose : `bool`, optional
        print verbose output, optional, default: `False`
    _SeriesClass : `type`, optional
        class object to use as the data holder for a single channel,
        default is :class:`~gwpy.timeseries.core.TimeSeries`

    Returns
    -------
    dict : :class:`~gwpy.timeseries.core.TimeSeriesDict`
        dict of (channel, `TimeSeries`) data pairs
    """
    if isinstance(channels, (unicode, str)):
        channels = channels.split(',')

    # open file
    if isinstance(framefile, CacheEntry):
        fp = framefile.path
    else:
        fp = framefile
    stream = frameCPP.IFrameFStream(fp)

    # interpolate frame epochs from CacheEntry
    # FIXME: update when new frameCPP is released
    nframe = 0 # int(stream.GetNumberOfFrames())
    if isinstance(framefile, CacheEntry) and nframe == 1:
        epochs = [framefile.segment[0]]
    else:
        epochs = None

    # load table of contents if needed
    if epochs is None or not type:
        toc = stream.GetTOC()
    # get list of frame epochs
    if epochs is None:
        epochs = toc.GTimeS
    # work out channel types
    if type:
        ctype = dict((str(channel), type) for channel in channels)
    else:
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
                ctype[name] = 'adc'
            elif name in procs:
                ctype[name] = 'proc'
            else:
                raise ValueError("Channel %s not found in frame table of "
                                 "contents" % name)

    # set output
    out = TimeSeriesDict()
    for channel in channels:
        name = str(channel)
        read_ = getattr(stream, 'ReadFr%sData' % ctype[name].title())
        ts = None
        i = 0
        while True:
            try:
                data = read_(i, name)
            except IndexError:
                break
            offset = data.GetTimeOffset()
            thisepoch = epochs[i] + offset
            for vect in data.data:
                arr = vect.GetDataArray()
                dx = vect.GetDim(0).dx
                if ts is None:
                    unit = vect.GetUnitY()
                    ts = _SeriesClass(arr, epoch=thisepoch, dx=dx, name=name,
                                      channel=channel, unit=unit, copy=True)
                else:
                    ts.append(arr)
            i += 1
        if ts is None:
            raise ValueError("Channel '%s' not found in frame '%s'"
                             % (str(channel), fp))
        else:
            out[channel] = ts

    return out


@with_import('frameCPP')
def read_timeseries(source, channel, **kwargs):
    """Read a `TimeSeries` of data from a gravitational-wave frame file

    Parameters
    ----------
    source : `str`, :class:`glue.lal.Cache`, `list`
        data source object, one of:

        - `str` : frame file path
        - :class:`glue.lal.Cache`, `list` : contiguous list of frame paths

    channel : :class:`~gwpy.detector.channel.Channel`, `str`
        data channel to read from frames

    See Also
    --------
    :func:`~gwpy.io.gwf.framecpp.read_timeseriesdict`
        for documentation on keyword arguments

    Returns
    -------
    data : :class:`~gwpy.timeseries.core.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    return read_timeseriesdict(source, [channel], **kwargs)[channel]


@with_import('frameCPP')
def read_statevectordict(source, channels, bitss=[], **kwargs):
    """Read a `StateVectorDict` of data from a gravitational-wave
    frame file.
    """
    kwargs.setdefault('_SeriesClass', StateVector)
    svd = StateVectorDict(read_timeseriesdict(source, channels, **kwargs))
    for (channel, bits) in zip(channels, bitss):
        svd[channel].bits = bits
    return svd


@with_import('frameCPP')
def read_statevector(source, channel, bits=None, **kwargs):
    """Read a `StateVector` of data from a gravitational-wave frame file

    Parameters
    ----------
    source : `str`, :class:`glue.lal.Cache`, `list`
        data source object, one of:

        - `str` : frame file path
        - :class:`glue.lal.Cache`, `list` : contiguous list of frame paths

    channel : :class:`~gwpy.detector.channel.Channel`, `str`
        data channel to read from frames
    bits : `list`
        ordered list of bit identifiers (names)

    See Also
    --------
    :func:`~gwpy.io.gwf.framecpp.read_timeseriesdict`
        for documentation on keyword arguments

    Returns
    -------
    data : :class:`~gwpy.timeseries.core.StateVector`
        a new `StateVector` containing the data read from disk
    """
    kwargs.setdefault('_SeriesClass', StateVector)
    sv = read_timeseries(source, channel, **kwargs)
    sv.bits = bits
    return sv


# register gwf reader first
try:
    import frameCPP
except ImportError:
    pass
else:
    try:
        register_identifier('gwf')
    except Exception as e:
        if not str(e).startswith('Identifier for format'):
            raise
    registry.register_reader('gwf', TimeSeriesDict, read_timeseriesdict,
                             force=True)
    registry.register_reader('gwf', StateVectorDict, read_statevectordict,
                             force=True)
    try:
        registry.register_reader('gwf', TimeSeries, read_timeseries, force=True)
    except:
        pass
    else:
        registry.register_reader('gwf', StateVector, read_statevector,
                                 force=True)

# register framecpp
registry.register_reader('framecpp', TimeSeriesDict, read_timeseriesdict)
registry.register_reader('framecpp', TimeSeries, read_timeseries)
registry.register_reader('framecpp', StateVectorDict, read_statevectordict)
registry.register_reader('framecpp', StateVector, read_statevector)

