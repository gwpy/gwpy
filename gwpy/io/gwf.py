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

"""Read and write gravitational-wave frame files.

The frame format is defined in LIGO-T970130 available from dcc.ligo.org.
"""

from __future__ import division

import decimal
import threading
import time
from math import ceil
from multiprocessing import (Process, Queue as ProcessQueue)
from Queue import Queue

from astropy.io import registry

from glue.lal import (Cache, CacheEntry)

from lal import Cache as LALCache
from lalframe import frread

from ..detector import Channel
from ..segments import Segment
from ..time import Time
from ..timeseries import (TimeSeries, TimeSeriesList)


class GWFInputThread(threading.Thread):
    """FIFO queued thread for reading a cache of GWF-format data files.
    """
    def __init__(self, source, channel, start=None, end=None,
                 datatype=None, verbose=False):
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
        self.datatype = datatype
        self.verbose = verbose

        # output attributes
        self.data = None

    def get_data(self):
        return self.data

    def run(self):
        self.data = read_gwf(self.source, self.channel, start=self.gpsstart,
                             end=self.gpsend, datatype=self.datatype,
                             verbose=self.verbose)


def read_gwf(framefile, channel, start=None, end=None, datatype=None,
             verbose=False):
    """Read a `TimeSeries` of data from a gravitational-wave frame file

    This method is a thin wrapper around `lalframe.frread.read_timeseries`
    and so can accept any input accepted by that function.

    Parameters
    ----------
    framefile : `str`, :class:`glue.lal.Cache`, :lalsuite:`LALCache`
        data source object, one of:

        - `str` : frame file path
        - :class:`glue.lal.Cache` : pure python cache object
        - :lalsuite:`LALCAche` : C-based cache object

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

    Returns
    -------
    data : :class:`~gwpy.timeseries.core.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    # parse input arguments
    if isinstance(framefile, CacheEntry):
        framefile = framefile.path
    elif isinstance(framefile, file):
        framefile = framefile.name
    if isinstance(channel, Channel):
        channel = channel.name
    if start and isinstance(start, Time):
        start = start.gps
    if end and isinstance(end, Time):
        end = end.gps
    if start and end:
        duration = float(end - start)
    elif end:
        raise ValueError("If `end` is given, `start` must also be given")
    else:
        duration = None
    lalts = frread.read_timeseries(framefile, channel, start=start,
                                   duration=duration, datatype=datatype,
                                   verbose=verbose)
    return TimeSeries.from_lal(lalts)


def read_cache_mp(cache, channel, start=None, end=None, datatype=None,
                  verbose=False, maxprocesses=1, minprocesssize=1):
    """Read a `TimeSeries` from a cache of GWF files using
    multiprocessing.

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
    maxprocesses : `int`, default: ``1``
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
    if isinstance(cache, (basestring, file)):
        cache = open_cache(cache)
    cache.sort(key=lambda ce: ce.segment[0])
    if start is not None and end is not None:
        span = Segment(start, end)
        cache = cache.sieve(segment=span)

    # single-process
    if maxprocesses == 1:
        return TimeSeries.read(cache, channel, format='gwf', start=start,
                               end=end, datatype=datatype, verbose=False)

    # define how to read each frame
    def _read(q, subcache):
        qs = float(max(start, subcache[0].segment[0]))
        qe = float(min(end, subcache[-1].segment[1]))
        q.put(TimeSeries.read(subcache, channel, format='gwf', start=qs,
                              end=qe, datatype=datatype, verbose=False))

    # separate cache into parts
    minprocesssize = max(int(ceil(len(cache)/maxprocesses)), minprocesssize)
    subcaches = [Cache(cache[i:i+minprocesssize]) for
                 i in range(0, len(cache), minprocesssize)]
    print len(subcaches)

    # start all processes
    queue = ProcessQueue()
    proclist = []
    for subcache in subcaches:
        process = Process(target=_read, args=(queue, subcache))
        process.daemon = True
        proclist.append(process)
        process.start()

    # get data and block
    out = TimeSeriesList(*(queue.get() for p in proclist))
    for process in proclist:
        process.join()

    # format and return
    out.sort(key=lambda ts: ts.epoch.gps)
    return out.join()


def read_cache_threaded(cache, channel, start=None, end=None, datatype=None,
                        verbose=False, maxthreads=10, minthreadsize=5):
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
    if isinstance(cache, (basestring, file)):
        cache = open_cache(cache)
    cache.sort(key=lambda ce: ce.segment[0])
    if start is not None and end is not None:
        span = Segment(start, end)
        cache = cache.sieve(segment=span)

    # define how to read each frame
    def _read(q, lcf):
        for subcache in lcf:
            qs = subcache[0].segment[0]
            qe = subcache[-1].segment[1]
            qs = start is not None and max(start, qs) or qs
            qe = end is not None and min(end, qe) or qe
            inthread = GWFInputThread(subcache, channel, qs, qe,
                                      datatype=datatype, verbose=verbose)
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

def open_cache(lcf):
    """Read a LAL-format cache file into memory as a
    :class:`glue.lal.Cache`.
    """
    if isinstance(lcf, file):
        return Cache.fromfile(lcf)
    else:
        with open(lcf, 'r') as f:
            return Cache.fromfile(f)


def identify_gwf(*args, **kwargs):
    """Determine an input file as written in GWF-format.
    """
    filename = args[1][0]
    if isinstance(filename, file):
        filename = filename.name
    elif isinstance(filename, CacheEntry):
        filename = filename.path
    if isinstance(filename, (str, unicode)) and filename.endswith('gwf'):
        return True
    else:
        return False


def identify_cache(*args, **kwargs):
    """Determine an input object as either a :class:`glue.lal.Cache` or
    a :lalsuite:`LALCache`.
    """
    cache = args[1][0]
    # identify cache object
    if isinstance(cache, (Cache, LALCache)):
        return True
    # identify open file
    elif isinstance(cache, file):
        cache = cache.name
    # identify string
    if (isinstance(cache, (unicode, str)) and
            (cache.endswith('.lcf') or cache.endswith('.cache'))):
        return True
    else:
        return False


registry.register_reader('gwf', TimeSeries, read_gwf, force=True)
registry.register_reader('lcf', TimeSeries, read_cache_mp, force=True)
registry.register_reader('cache', TimeSeries, read_cache_mp, force=True)
registry.register_reader('lcfmp', TimeSeries, read_cache_mp, force=True)
registry.register_reader('lcfth', TimeSeries, read_cache_threaded, force=True)
registry.register_identifier('gwf', TimeSeries, identify_gwf)
registry.register_identifier('lcf', TimeSeries, identify_cache)
