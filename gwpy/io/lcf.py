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
import threading
from decimal import _Infinity as infinity
from math import ceil
from multiprocessing import (Process, Queue as ProcessQueue)
from Queue import Queue

from astropy.io import registry

try:
    from glue.lal import (Cache, CacheEntry)
except ImportError:
    HASGLUE = False
else:
    HASGLUE = True

try:
    from lal import Cache as LALCache
except ImportError:
    HASLAL = False
else:
    HASLAL = True

from ..segments import Segment
from ..timeseries import (TimeSeries, TimeSeriesList, TimeSeriesDict,
                          StateVector, StateVectorDict)


class GWFInputThread(threading.Thread):
    """FIFO queued thread for reading a cache of GWF-format data files.
    """
    def __init__(self, source, channel, start=None, end=None, format=None,
                 target=TimeSeries, **kwargs):
        """Define a new input thread.

        This constructor does not start the thread. To start the thread
        run the :meth:`~GWFInputThread.start` method.
        """
        super(GWFInputThread, self).__init__()

        # input attributes
        self.source = source
        self.channel = channel
        self.gpsstart = float(start)
        self.gpsend = float(end)
        if format is None and isinstance(source, (unicode, str)):
            self.format = os.path.splitext(source)[1][1:]
        else:
            self.format = format
        self.cls = target
        self.kwargs = kwargs

        # output attributes
        self.data = None

    def get_data(self):
        return self.data

    def run(self):
        self.data = self.cls.read(self.source, self.channel,
                                  format=self.format, start=self.gpsstart,
                                  end=self.gpsend, **self.kwargs)


def read_cache_mp(cache, channel, start=None, end=None, resample=None,
                  maxprocesses=1, minprocesssize=1, maxprocesssize=None,
                  **kwargs):
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
    maxprocesses : `int`, default: ``1``
        maximum number of independent frame reading processes, default
        is set to single-process file reading.
    minprocesssize : `int`, default: ``1``
        minimum number of frames to pass to a single process, default is
        to maximally separate the cache.
    maxprocesssize : `int`, default: ``1``
        maximum number of frames to pass to a single process, default is
        to use as many proccesses as possible (up to ``maxprocesses``)

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

    format_ = os.path.splitext(cache[0].path)[1][1:]

    # single-process
    if maxprocesses == 1:
        return cls.read(cache, channel, format=format_, start=start, end=end,
                        resample=resample, **kwargs)

    # define how to read each frame
    def _read(q, subcache):
        qs = float(max(start or -1, float(subcache[0].segment[0])))
        qe = float(min(end or infinity, float(subcache[-1].segment[1])))
        q.put(cls.read(subcache, channel, format=format_, start=qs, end=qe,
                       resample=resample, **kwargs))

    # separate cache into parts
    fperproc = int(ceil(len(cache) / maxprocesses))
    if fperproc < minprocesssize:
        fperproc = minprocesssize
    if maxprocesssize is not None and fperproc > maxprocessize:
        fperproc = maxprocesssize
    subcaches = [Cache(cache[i:i+fperproc]) for
                 i in range(0, len(cache), fperproc)]

    # start all processes
    queue = ProcessQueue(maxprocesses)
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
        if len(channels):
            data.sort(key=lambda tsd: tsd.values()[0].epoch.gps)
        out = cls()
        while len(data):
            tsd = data.pop(0)
            out.append(tsd)
            del tsd
        return out
    else:
        out = TimeSeriesList(*data)
        out.sort(key=lambda ts: ts.epoch.gps)
        return out.join()


def read_cache_threaded(cache, channel, start=None, end=None, format=None,
                        maxthreads=10, minthreadsize=5, **kwargs):
    """Read a `TimeSeries` from a cache of GWF files using threading.

    .. warning::

        This function has not been rigorously tested, so the defaults
        might not be ideal. If you want to use it, and can recommend
        improvements, please send them along on github.

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
    datatype : `type`, `numpy.dtype`, `str`, optional
        identifier for desired output data type
    verbose : `bool`, optional
        print verbose output
    maxthreads : `int`, default: ``10``
        maximum number of independent frame reading processes, default
        is set to single-process file reading.
    minprocesssize : `int`, default: ``1``
        minimum number of frames to pass to a single process, default is
        to maximally separate the cache.

    Notes
    -----
    The number of independent processes spawned by this function can be
    calculated as ``min(maxprocesses, len(cache)//minprocesssize)``.

    Returns
    -------
    data : :class:`~gwpy.timeseries.core.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    # open cache from file if given
    if isinstance(cache, (unicode, str, file)):
        cache = open_cache(cache)
    cache.sort(key=lambda ce: ce.segment[0])
    if start is not None and end is not None:
        span = Segment(start, end)
        cache = cache.sieve(segment=span)

    if format is None and len(cache):
        format_ = os.path.splitext(cache[0].path)[1][1:]
    else:
        format_ = format

    # define how to read each frame
    def _read(q, lcf):
        for subcache in lcf:
            qs = subcache[0].segment[0]
            qe = subcache[-1].segment[1]
            qs = start is not None and max(start, qs) or qs
            qe = end is not None and min(end, qe) or qe
            inthread = GWFInputThread(subcache, channel, qs, qe,
                                      format=format_, **kwargs)
            inthread.start()
            q.put(inthread, True)

    # define how to collect results
    complete = TimeSeriesList()

    def _access(q, numthreads):
        while len(complete) < numthreads:
            inthread = q.get(True)
            inthread.join()
            complete.append(inthread.get_data())

    # optimise threads
    subcaches = [Cache(cache[i:i+minthreadsize]) for
                 i in range(0, len(cache), minthreadsize)]

    # set up threads
    q = Queue(maxthreads)
    readthread = threading.Thread(target=_read, args=(q, subcaches))
    accessthread = threading.Thread(target=_access, args=(q, len(subcaches)))

    # read all frames
    readthread.start()
    accessthread.start()
    readthread.join()
    accessthread.join()

    # join the frames together
    complete.sort(key=lambda ts: ts.epoch.gps)
    return complete.join()


def read_state_cache_mp(*args, **kwargs):
    kwargs.setdefault('target', StateVector)
    return read_cache_mp(*args, **kwargs)

def read_dict_cache_mp(*args, **kwargs):
    kwargs.setdefault('target', TimeSeriesDict)
    return read_cache_mp(*args, **kwargs)

def read_state_dict_cache_mp(*args, **kwargs):
    kwargs.setdefault('target', StateVectorDict)
    return read_cache_mp(*args, **kwargs)


def open_cache(lcf):
    """Read a LAL-format cache file into memory as a
    :class:`glue.lal.Cache`.
    """
    if not HASGLUE:
        raise ImportError("No module name glue.lal")
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
    if ((HASGLUE and isinstance(cacheobj, Cache)) or
            (HASLAL and isinstance(cacheobj, LALCache))):
        return True
    else:
        return False


registry.register_reader('lcf', TimeSeries, read_cache_mp, force=True)
registry.register_reader('cache', TimeSeries, read_cache_mp, force=True)
registry.register_reader('lcfmp', TimeSeries, read_cache_mp, force=True)
registry.register_reader('lcfth', TimeSeries, read_cache_threaded, force=True)
registry.register_identifier('lcf', TimeSeries, identify_cache_file)
registry.register_identifier('cache', TimeSeries, identify_cache)

# duplicate for state-vector
registry.register_reader('lcf', StateVector, read_state_cache_mp, force=True)
registry.register_reader('cache', StateVector, read_state_cache_mp, force=True)
registry.register_reader('lcfmp', StateVector, read_state_cache_mp, force=True)
registry.register_identifier('lcf', StateVector, identify_cache_file)
registry.register_identifier('cache', StateVector, identify_cache)

# TimeSeriesDict
registry.register_reader('lcf', TimeSeriesDict, read_dict_cache_mp)
registry.register_reader('cache', TimeSeriesDict, read_dict_cache_mp)
registry.register_reader('lcfmp', TimeSeriesDict, read_dict_cache_mp)
registry.register_identifier('lcf', TimeSeriesDict, identify_cache)

# StateVectorDict
registry.register_reader('lcf', StateVectorDict, read_state_dict_cache_mp)
registry.register_reader('cache', StateVectorDict, read_state_dict_cache_mp)
registry.register_reader('lcfmp', StateVectorDict, read_state_dict_cache_mp)
registry.register_identifier('lcf', StateVectorDict, identify_cache)
