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
"""

from __future__ import division

from ....io.cache import (CacheEntry, file_list)
from ....time import to_gps
from ....utils import (import_method_dependency, with_import)
from ... import (TimeSeries, TimeSeriesDict)
from . import channel_dict_kwarg

DEPENDS = 'lalframe.frread'


@with_import(DEPENDS)
def read_timeseriesdict(source, channels, start=None, end=None, dtype=None,
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
    lal = import_method_dependency('lal.lal')
    from gwpy.utils.lal import to_lal_ligotimegps
    # parse input arguments (astropy insists on opening files)
    if isinstance(source, file):
        source = source.name
    filelist = file_list(source)
    try:
        frametype = CacheEntry.from_T050017(filelist[0]).description
    except ValueError:
        frametype = None
    # set times
    if start is not None:
        start = to_lal_ligotimegps(start)
    if end is not None:
        end = to_lal_ligotimegps(end)
    if start and end:
        duration = float(end - start)
    elif end:
        raise ValueError("If `end` is given, `start` must also be given")
    else:
        duration = None
    # parse resampling
    resample = channel_dict_kwarg(resample, channels, (int,))
    if resample is None:
        raise ValueError("Cannot parse resample request, please review "
                         "documentation for that argument")
    # parse dtype
    dtype = channel_dict_kwarg(dtype, channels, (str, type))
    if dtype is None:
        raise ValueError("Cannot parse dtype request, please review "
                         "documentation for that argument")

    # read data
    try:
        laldata = frread.read_timeseries(source, map(str, channels),
                                         start=start, duration=duration,
                                         verbose=verbose)
    # if old version of lalframe.frread
    except TypeError:
        laldata = [frread.read_timeseries(source, str(c), start=start,
                                          duration=duration, verbose=verbose)
                   for c in channels]
    # convert to native objects and return
    out = TimeSeriesDict()
    for channel, lalts in zip(channels, laldata):
        ts = _SeriesClass.from_lal(lalts)
        ts.channel.frametype = frametype
        if channel in dtype:
            ts = ts.astype(dtype[channel])
        if channel in resample:
            ts = ts.resample(resample[channel])
        out[channel] = ts
    return out
