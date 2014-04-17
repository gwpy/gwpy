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
from glue.ligolw.lsctables import TableByName
from glue.ligolw.table import (StripTableName as strip_table_name)

from ...io.cache import (open_cache, identify_cache, identify_cache_file)


def read_cache_single(cache, tablename, columns=None, **kwargs):
    """Read a `Table` of data from a `Cache` of files.
    """
    # convert to list of files (basically to fool auto-ID)
    files = cache.pfnlist()
    tablename = strip_table_name(tablename)

    TableClass = TableByName[tablename]

    return TableClass.read(files, columns=columns, **kwargs)


def read_cache(cache, tablename, columns=None, nproc=1,
               sort=None, **kwargs):
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

        .. warning::

           When multiprocessing, the ordering of rows in the output
           `Table` is effectively random. Users can supply a
           `lambda`-style ``sort`` key if ordering is important.

    sort : `callable`, optional
        callable function to pass to the
        :meth:`~glue.ligolw.table.Table.sort` method in order to sort
        the output table.

    Notes
    -----
    The number of independent processes spawned by this function can be
    calculated as ``min(maxprocesses, len(cache)//minprocesssize)``.

    Returns
    -------
    data : :class:`~gwpy.timeseries.core.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    if isinstance(cache, (file, unicode, str)):
        cache = open_cache(cache)
    if isinstance(cache, Cache):
        cache.sort(key=lambda ce: ce.segment[0])

    # force one file per process minimum
    nproc = min(nproc, len(cache))

    # work out the underlying file type
    tablename = strip_table_name(tablename)
    TableClass = TableByName[strip_table_name(tablename)]
    try:
        kwargs.setdefault(
            'format', registry._get_valid_format('read', TableClass, None,
                                                 None, (cache[0],), {}))
    except IndexError:
        kwargs.setdefault('format', 'ligolw')
    except Exception:
        if 'format' not in kwargs:
            raise

    # single-process
    if nproc <= 1:
        out = read_cache_single(cache, tablename, columns=columns,
                                **kwargs)
        if sort:
            out.sort(key=sort)
        return out

    # define how to read each sub-cache
    def _read(q, sc):
        try:
            q.put(read_cache_single(sc, tablename, columns=columns,
                                    **kwargs))
        except Exception as e:
            q.put(e)

    # separate cache into parts
    fperproc = int(ceil(len(cache) / nproc))
    subcaches = [cache.__class__(cache[i:i+fperproc]) for
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
    data = []
    for i in range(len(proclist)):
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        data.append(result)
    for process in proclist:
        process.join()

    # coalesce and return
    out = data[0]
    extend = out.extend
    for table_ in data[1:]:
        extend(table_)
    if sort:
        out.sort(key=sort)
    return out


def _read_factory(table_):
    """Define a custom function to read this table from a cache.
    """
    def _read(f, **kwargs):
        return read_cache(f, table_.tableName, **kwargs)
    return _read


# register cache reading for all lsctables
for table in TableByName.itervalues():
    registry.register_reader('lcf', table, _read_factory(table))
    registry.register_reader('cache', table, _read_factory(table))
    registry.register_identifier('lcf', table, identify_cache_file)
    registry.register_identifier('cache', table, identify_cache)
