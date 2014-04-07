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

from math import ceil
from multiprocessing import (Process, Queue as ProcessQueue)

from astropy.io import registry

from glue.lal import Cache
from glue.ligolw import lsctables
from glue.ligolw.table import (StripTableName as strip_table_name)

from .. import _TABLES
from ...io.cache import (open_cache, identify_cache, identify_cache_file)
from ...utils import gprint
from ...segments import (Segment, SegmentList)


def read_cache_single(cache, tablename, columns=None, verbose=False, **kwargs):
    """Read a `Table` of data from a `Cache` of files.
    """
    # open cache from file if given
    if isinstance(cache, (unicode, str, file)):
        cache = open_cache(cache)
    nfiles = len(cache)
    tablename = strip_table_name(tablename)

    if verbose:
        if not isinstance(verbose, (unicode, str)):
            verbose = ''

        def status(pos, percent=None, final=False):
            if percent is None:
                percent = pos/nfiles * 100
            gprint("%sReading %s from files... %d/%d (%.1f%%)"
                   % (verbose, tablename, pos, nfiles, percent),
                   end=final and '\n' or '\r')

    # format output object
    TableClass = lsctables.TableByName[tablename]

    # return no files
    if nfiles == 0:
        if verbose is not False:
            status(0, 100)
        return lsctables.New(TableClass, columns=columns)
    elif verbose is not False:
        status(0)

    # otherwise read first file
    out = TableClass.read(cache[0], columns=columns, **kwargs)
    if verbose is not False:
        status(1)
    extend = out.extend

    # then read other files
    for i, entry in enumerate(cache[1:]):
        extend(TableClass.read(entry, columns=columns, **kwargs))
        if verbose is not False:
            status(i+2)
    if verbose is not False:
        status(nfiles, final=True)

    return out


def read_cache(cache, tablename, columns=None, nproc=1, **kwargs):
    """Read a `Table` of data from a `Cache`.

    The inner-workings are agnostic of data-type, but can only handle a
    single data type at a time.

    Parameters
    ----------
    cache : :class:`glue.lal.Cache`, `str`
        cache of files files, or path to a LAL-format cache file
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
    verbose = kwargs.pop('verbose', False)

    # open cache from file if given
    if isinstance(cache, (unicode, str, file)):
        cache = open_cache(cache)

    # use cache to get start end times
    cache.sort(key=lambda ce: ce.segment[0])

    # force one file per process minimum
    nproc = min(nproc, len(cache))

    # single-process
    if nproc <= 1:
        return read_cache_single(cache, tablename, columns=columns,
                                 verbose=verbose, **kwargs)

    # define how to read each sub-cache
    def _read(q, pstart, pend):
        subcache = cache.sieve(segment=Segment(pstart, pend))
        q.put(read_cache_single(subcache, tablename, columns=columns,
                                **kwargs))

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
    for process in proclist:
        process.join()

    # coalesce and return
    out = data[0]
    extend = out.extend
    for table_ in data[1:]:
        extend(table_)
    return out


def _read_factory(table_):
    """Define a custom function to read this table from a cache.
    """
    def _read(f, **kwargs):
        return read_cache(f, table_.tableName, **kwargs)
    return _read


# register cache reading for all lsctables
for name, table in _TABLES.iteritems():
    registry.register_reader('lcf', table, _read_factory(table))
    registry.register_reader('cache', table, _read_factory(table))
    registry.register_identifier('lcf', table, identify_cache_file)
    registry.register_identifier('cache', table, identify_cache)
