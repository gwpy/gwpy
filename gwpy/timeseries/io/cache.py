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

"""Read gravitational-wave data from a cache of files
"""

from __future__ import division

import os
import warnings
from math import ceil
from multiprocessing import (Process, Queue as ProcessQueue)

from glue.lal import Cache

from ...io import registry
from ...io.cache import (cache_segments, open_cache)
from .. import (TimeSeries, TimeSeriesList, TimeSeriesDict,
                StateVector, StateVectorList, StateVectorDict)

# set maximum number of channels with which to still use lalframe
MAX_LALFRAME_CHANNELS = 4


def read_cache(cache, channel, start=None, end=None, resample=None,
               gap=None, pad=None, nproc=1, format=None, **kwargs):
    """Read a `TimeSeries` from a cache of data files using
    multiprocessing.

    The inner-workings are agnostic of data-type, but can only handle a
    single data type at a time.

    Parameters
    ----------
    cache : :class:`glue.lal.Cache`, `str`
        cache of GWF frame files, or path to a LAL-format cache file
        on disk
    channel : :class:`~gwpy.detector.channel.Channel`, `str`
        data channel to read from frames
    start : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        end GPS time of desired data
    resample : `float`, optional
        rate (samples per second) to resample
    format : `str`, optional
        name of data file format, e.g. ``gwf`` or ``hdf``.
    nproc : `int`, default: ``1``
        maximum number of independent frame reading processes, default
        is set to single-process file reading.
    gap : `str`, optional
        how to handle gaps in the cache, one of

        - 'ignore': do nothing, let the undelying reader method handle it
        - 'warn': do nothing except print a warning to the screen
        - 'raise': raise an exception upon finding a gap (default)
        - 'pad': insert a value to fill the gaps

    pad : `float`, optional
        value with which to fill gaps in the source data, only used if
        gap is not given, or `gap='pad'` is given

    Notes
    -----
    The number of independent processes spawned by this function can be
    calculated as ``min(maxprocesses, len(cache)//minprocesssize)``.

    Returns
    -------
    data : :class:`~gwpy.timeseries.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    from gwpy.segments import (Segment, SegmentList)

    cls = kwargs.pop('target', TimeSeries)
    # open cache from file if given
    if isinstance(cache, (unicode, str, file)):
        cache = open_cache(cache)

    # fudge empty cache
    if len(cache) == 0:
        return cls([], channel=channel, epoch=start)

    # use cache to get start end times
    cache.sort(key=lambda ce: ce.segment[0])
    if start is None:
        start = cache[0].segment[0]
    if end is None:
        end = cache[-1].segment[1]

    # get span
    span = Segment(start, end)
    if cls not in (StateVector, StateVectorDict) and resample:
        cache = cache.sieve(segment=span.protract(8))
    else:
        cache = cache.sieve(segment=span)
    cspan = Segment(cache[0].segment[0], cache[-1].segment[1])

    # check for gaps
    if gap is None and pad is not None:
        gap = 'pad'
    elif gap is None:
        gap = 'raise'
    segs = cache_segments(cache, on_missing='ignore') & SegmentList([span])
    if len(segs) != 1 and gap.lower() == 'ignore' or gap.lower() == 'pad':
        pass
    elif len(segs) != 1:
        gaps = SegmentList([cspan]) - segs
        msg = ("The cache given to %s.read has gaps in it in the "
               "following segments:\n    %s"
               % (cls.__name__, '\n    '.join(map(str, gaps))))
        if gap.lower() == 'warn':
            warnings.warn(msg)
        else:
            raise ValueError(msg)
        segs = type(segs)([span])

    # if reading a small number of channels, try to use lalframe, its faster
    if format is None and (
            isinstance(channel, str) or (isinstance(channel, (list, tuple)) and
            len(channel) <= MAX_LALFRAME_CHANNELS)):
        try:
            from lalframe import frread
        except ImportError:
            format = 'gwf'
        else:
            kwargs.pop('type', None)
            format = 'lalframe'
    # otherwise use the file extension as the format
    elif format is None:
        format = os.path.splitext(cache[0].path)[1][1:]

    # -- process multiple cache segments --------
    # this entry point loops this method for each segment

    if len(segs) > 1:
        out = None
        for seg in segs:
            new = read_cache(cache, channel, start=seg[0], end=seg[1],
                             resample=resample, nproc=nproc, format=format,
                             target=cls, **kwargs)
            if out is None:
                out = new
            else:
                out.append(new, gap='pad', pad=pad)
        return out

    # -- process single cache segment

    # force one frame per process minimum
    nproc = min(nproc, len(cache))

    # single-process
    if nproc <= 1:
        return cls.read(cache, channel, format=format, start=start, end=end,
                        resample=resample, **kwargs)

    # define how to read each frame
    def _read(q, pstart, pend):
        try:
            # don't go beyond the requested limits
            pstart = float(max(start, pstart))
            pend = float(min(end, pend))
            # if resampling TimeSeries, pad by 8 seconds inside cache limits
            if cls not in (StateVector, StateVectorDict) and resample:
                cstart = float(max(cspan[0], pstart - 8))
                subcache = cache.sieve(segment=Segment(cstart, pend))
                out = cls.read(subcache, channel, format=format, start=cstart,
                               end=pend, resample=None, **kwargs)
                out = out.resample(resample)
                q.put(out.crop(pstart, pend))
            else:
                subcache = cache.sieve(segment=Segment(pstart, pend))
                q.put(cls.read(subcache, channel, format=format, start=pstart,
                               end=pend, resample=resample, **kwargs))
        except Exception as e:
            q.put(e)

    # separate cache into parts
    fperproc = int(ceil(len(cache) / nproc))
    subcaches = [Cache(cache[i:i+fperproc]) for
                 i in range(0, len(cache), fperproc)]
    subsegments = SegmentList([Segment(c[0].segment[0], c[-1].segment[1])
                               for c in subcaches])

    # start all processes
    queue = ProcessQueue(nproc)
    proclist = []
    for subseg in subsegments:
        process = Process(target=_read, args=(queue, subseg[0], subseg[1]))
        process.daemon = True
        proclist.append(process)
        process.start()

    # get data and block
    data = [queue.get() for p in proclist]
    for result in data:
        process.join()
        if isinstance(result, Exception):
            raise result

    # format and return
    if issubclass(cls, dict):
        try:
            data.sort(key=lambda tsd: tsd.values()[0].epoch.gps)
        except IndexError:
            pass
        out = cls()
        while len(data):
            tsd = data.pop(0)
            out.append(tsd)
            del tsd
        return out
    else:
        if cls in (TimeSeries, TimeSeriesDict):
            out = TimeSeriesList(*data)
        else:
            out = StateVectorList(*data)
        out.sort(key=lambda ts: ts.epoch.gps)
        ts = out.join(gap=gap)
        return ts


def read_state_cache(*args, **kwargs):
    kwargs.setdefault('target', StateVector)
    return read_cache(*args, **kwargs)


def read_dict_cache(*args, **kwargs):
    kwargs.setdefault('target', TimeSeriesDict)
    return read_cache(*args, **kwargs)


def read_state_dict_cache(*args, **kwargs):
    kwargs.setdefault('target', StateVectorDict)
    return read_cache(*args, **kwargs)
