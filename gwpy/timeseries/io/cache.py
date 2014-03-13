# coding=utf-8
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
from decimal import _Infinity as infinity
from math import ceil
from multiprocessing import (Process, Queue as ProcessQueue)

from astropy.io import registry

from glue.lal import Cache

from ...segments import Segment
from .. import (TimeSeries, TimeSeriesList, TimeSeriesDict,
                StateVector, StateVectorDict)


def read_cache(cache, channel, start=None, end=None, resample=None,
               nproc=1, **kwargs):
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
    start : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        end GPS time of desired data
    resample : `float`, optional
        rate (samples per second) to resample
    format : `str`, optional
        name of data file format, e.g. ``gwf`` or ``hdf``.
    nproc : `int`, default: ``1``
        maximum number of independent frame reading processes, default
        is set to single-process file reading.

    Notes
    -----
    The number of independent processes spawned by this function can be
    calculated as ``min(maxprocesses, len(cache)//minprocesssize)``.

    Returns
    -------
    data : :class:`~gwpy.timeseries.core.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    cls = kwargs.pop('target', TimeSeries)
    # open cache from file if given
    if isinstance(cache, (unicode, str, file)):
        cache = open_cache(cache)
    cache.sort(key=lambda ce: ce.segment[0])
    if start is not None and end is not None:
        span = Segment(start, end)
        cache = cache.sieve(segment=span)

    if len(cache) == 0:
        return cls([], channel=channel, epoch=start)

    if isinstance(channel, (list, tuple)) and len(channel) == 1:
        try:
            from lalframe import frread
        except ImportError:
            format_ = 'gwf'
        else:
            kwargs.pop('type', None)
            format_ = 'lalframe'
    else:
        format_ = os.path.splitext(cache[0].path)[1][1:]

    # single-process
    if nproc == 1:
        return cls.read(cache, channel, format=format_, start=start, end=end,
                        resample=resample, **kwargs)

    # define how to read each frame
    def _read(q, subcache):
        qs = float(max(start or -1, float(subcache[0].segment[0])))
        qe = float(min(end or infinity, float(subcache[-1].segment[1])))
        if cls in (StateVector, StateVectorDict):
            q.put(cls.read(subcache, channel, format=format_, start=qs, end=qe,
                           resample=resample, **kwargs))
        else:
            q.put(cls.read(subcache, channel, format=format_, start=qs, end=qe,
                           resample=None, **kwargs))

    # separate cache into parts
    fperproc = int(ceil(len(cache) / nproc))
    subcaches = [Cache(cache[i:i+fperproc]) for
                 i in range(0, len(cache), fperproc)]

    # start all processes
    queue = ProcessQueue(nproc)
    proclist = []
    for subcache in subcaches:
        process = Process(target=_read, args=(queue, subcache))
        process.daemon = True
        proclist.append(process)
        process.start()

    # get data and block
    data = [queue.get() for p in proclist]
    for process in proclist:
        process.join()

    # format and return
    if issubclass(cls, dict):
        if isinstance(channel, (unicode, str)):
            channels = channel.split(',')
        else:
            channels = channel
        try:
            data.sort(key=lambda tsd: tsd.values()[0].epoch.gps)
        except IndexError:
            pass
        out = cls()
        while len(data):
            tsd = data.pop(0)
            out.append(tsd)
            del tsd
        if not isinstance(out, StateVectorDict) and resample:
            out.resample(resample)
        return out
    else:
        out = TimeSeriesList(*data)
        out.sort(key=lambda ts: ts.epoch.gps)
        ts = out.join()
        if not isinstance(ts, StateVector) and resample:
           ts = ts.resample(resample)
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


def open_cache(lcf):
    """Read a LAL-format cache file into memory as a
    :class:`glue.lal.Cache`.
    """
    if isinstance(lcf, file):
        return Cache.fromfile(lcf)
    else:
        with open(lcf, 'r') as f:
            return Cache.fromfile(f)


def identify_cache_file(*args, **kwargs):
    """Determine an input object as either a LAL-format cache file.
    """
    cachefile = args[1]
    if isinstance(cachefile, file):
        cachefile = cachefile.name
    # identify string
    if (isinstance(cachefile, (unicode, str)) and
            (cachefile.endswith('.lcf') or cachefile.endswith('.cache'))):
        return True
        # identify cache object
    else:
        return False


def identify_cache(*args, **kwargs):
    """Determine an input object as a :class:`glue.lal.Cache` or a
    :lalsuite:`LALCache`.
    """
    cacheobj = args[3]
    if isinstance(cacheobj, Cache):
            return True
    try:
        from lal import Cache as LALCache
    except ImportError:
        pass
    else:
        if isinstance(cacheobj, LALCache):
            return True
    return False


registry.register_reader('lcf', TimeSeries, read_cache)
registry.register_reader('cache', TimeSeries, read_cache)
registry.register_identifier('lcf', TimeSeries, identify_cache_file)
registry.register_identifier('cache', TimeSeries, identify_cache)

# duplicate for state-vector
registry.register_reader('lcf', StateVector, read_state_cache)
registry.register_reader('cache', StateVector, read_state_cache)
registry.register_identifier('lcf', StateVector, identify_cache_file)
registry.register_identifier('cache', StateVector, identify_cache)

# TimeSeriesDict
registry.register_reader('lcf', TimeSeriesDict, read_dict_cache)
registry.register_reader('cache', TimeSeriesDict, read_dict_cache)
registry.register_reader('lcfmp', TimeSeriesDict, read_dict_cache)
registry.register_identifier('lcf', TimeSeriesDict, identify_cache_file)
registry.register_identifier('cache', TimeSeriesDict, identify_cache)

# StateVectorDict
registry.register_reader('lcf', StateVectorDict, read_state_dict_cache)
registry.register_reader('cache', StateVectorDict, read_state_dict_cache)
registry.register_reader('lcfmp', StateVectorDict, read_state_dict_cache)
registry.register_identifier('lcf', StateVectorDict, identify_cache_file)
registry.register_identifier('cache', StateVectorDict, identify_cache)
