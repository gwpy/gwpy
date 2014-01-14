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

try:
    import frameCPP
except ImportError:
    raise ImportError("No module named frameCPP. lalframe or frameCPP are "
                      "required in order to read data from GWF-format "
                      "frame files.")

from astropy.io import registry

from glue.lal import (Cache, CacheEntry)

from ... import version
from ...utils import gprint
from ...timeseries import (TimeSeries, TimeSeriesDict, StateVector)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


def read_timeseriesdict(source, channels, start=None, end=None, type=None,
                        verbose=False):
    """Read the data for a list of channels from a GWF data source.

    Parameters
    ----------
    source : `str`, :class:`glue.lal.Cache`, `list`
        data source object, one of:

        - `str` : frame file path
        - :class:`glue.lal.Cache`, `list` : contiguous list of frame paths

    channels : `list`
        list of channel names (or `Channel` objects) to read from frame
    start : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        end GPS time of desired data
    channeltype : `str`
        type of channel, one of ``adc`` or ``proc``
    verbose : `bool`, optional
        print verbose output

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
        filelist = [source.path]
    elif isinstance(source, Cache):
        source.sort(key=lambda e: e.segment[0])
        filelist = source.pfnlist()
    else:
        filelist = list(source)
    # read each individually and append
    N = len(filelist)
    if verbose:
        if not isinstance(verbose, (unicode, str)):
            verbose = ''
        gprint("%sReading %d channels from frames... 0/%d (0.00%%)\r"
               % (verbose, len(channels), N), end='')
    out = TimeSeriesDict()
    for i,fp in enumerate(filelist):
        out.append(_read_frame(fp, channels, start=start, end=end, type=type,
                               verbose=verbose))
        if verbose is not False:
            gprint("%sReading %d channels from frames... %d/%d (%.1f%%)\r"
                   % (verbose, len(channels), i, N, i/N * 100), end='')
    if verbose is not False:
        gprint("%sReading %d channels from frames... %d/%d (100%%)"
               % (verbose, len(channels), N, N))
    return out



def _read_frame(framefile, channels, start=None, end=None, type=None,
                verbose=False):
    """Internal function to read data from a single frame.

    All users should be using the wrapper `read_timeseriesdict`.

    Returns
    -------
    dict : :class:`~gwpy.timeseries.core.TimeSeriesDict`
        dict of (channel, `TimeSeries`) data pairs
    """
    if isinstance(channels, (unicode, str)):
        channels = channels.split(',')
    # open file
    stream = frameCPP.IFrameFStream(framefile)
    # read table of contents
    toc = stream.GetTOC()
    nframes = toc.GetNFrame()
    epochs = toc.GTimeS
    # read channel lists: XXX: this needs optimised, most of time taken is
    #                          building the channel lists
    if not type:
        try:
            adcs = toc.GetADC().keys()
        except AttributeError:
            adcs = []
        try:
            procs = toc.GetProc().keys()
        except AttributeError:
            procs = []

    out = TimeSeriesDict()
    for channel in channels:
        name = str(channel)
        if type:
            read_ = getattr(stream, 'ReadFr%sData' % type.title())
        else:
            read_ = (name in adcs and stream.ReadFrAdcData or
                     name in procs and stream.ReadFrProcData or None)
        if read_ is None:
            raise ValueError("Channel %s not found in frame table of contents"
                             % name)
        ts = None
        for i in range(nframes):
            data = read_(i, name)
            offset = data.GetTimeOffset()
            fs = data.GetSampleRate()
            for vect in data.data:
                arr = vect.GetDataArray()
                if ts is None:
                    unit = vect.GetUnitY()
                    ts = TimeSeries(arr, epoch=epochs[i] + offset,
                                    sample_rate=fs, name=name, channel=channel,
                                    unit=unit)
                else:
                    ts.append(arr)
        if ts is not None:
            out[channel] = ts

    return out


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


def read_statevector(source, channel, bitmask=[], **kwargs):
    """Read a `StateVector` of data from a gravitational-wave frame file

    Parameters
    ----------
    source : `str`, :class:`glue.lal.Cache`, `list`
        data source object, one of:

        - `str` : frame file path
        - :class:`glue.lal.Cache`, `list` : contiguous list of frame paths

    channel : :class:`~gwpy.detector.channel.Channel`, `str`
        data channel to read from frames
    bitmask : `list`
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
    sv = read_timeseries(source, channel, **kwargs).view(StateVector)
    sv.bitmask = bitmask
    return sv

registry.register_reader('framecpp', TimeSeriesDict, read_timeseriesdict)
registry.register_reader('framecpp', TimeSeries, read_timeseries)
registry.register_reader('framecpp', StateVector, read_statevector)

# force this as the 'gwf' reader
registry.register_reader('gwf', TimeSeriesDict, read_timeseriesdict, force=True)
registry.register_reader('gwf', TimeSeries, read_timeseries, force=True)
registry.register_reader('gwf', StateVector, read_statevector, force=True)
