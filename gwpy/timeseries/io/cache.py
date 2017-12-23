# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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

"""Multi-processed I/O utilities for `TimeSeries`
"""

from math import ceil

from six import string_types

from astropy.io.registry import read as io_read

from ...io.registry import get_read_format
from ...io.cache import (read_cache, cache_segments, FILE_LIKE)
from ...segments import (Segment, SegmentList)
from ...utils.mp import multiprocess_with_queues
from ..core import TimeSeriesBaseList
from ..timeseries import TimeSeries
from ..statevector import (StateVector, StateVectorDict)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def read_from_cache(cache, channel, start=None, end=None, resample=None,
                    gap=None, pad=None, nproc=1, format=None, verbose=False,
                    **kwargs):
    """Read a `TimeSeries` from a cache of data files using multiprocessing

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
    data : `~gwpy.timeseries.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    cls = kwargs.pop('target', TimeSeries)

    # open cache file
    if isinstance(cache, FILE_LIKE + string_types):
        cache = read_cache(cache)
    cache = type(cache)(cache)
    cache.sort(key=lambda e: e.segment)

    # get input format
    if kwargs.get('format', None) is None:
        kwargs['format'] = get_read_format(cls, cache[0].path, (), {})

    # handle resampling as an overlap for TimeSeries
    if not resample or cls in (StateVector, StateVectorDict):
        overlap = 0
    else:
        overlap = 4

    # get timing
    if start is None:  # start time of earliest file
        start = cache[0].segment[0]
    if end is None:  # end time of latest file
        end = cache[-1].segment[-1]
    span = Segment(start, end)
    cache = cache.sieve(segment=span.protract(overlap))
    cspan = Segment(cache[0].segment[0], cache[-1].segment[1])

    # define segments for multiprocessing
    segs = get_mp_cache_segments(cache, nproc, span, overlap=overlap)

    # define reader for each segment
    def _read_segment(segment):
        try:
            if segment[0] is None:
                pseg = segment
                subcache = cache
            else:
                pseg = segment.protract(overlap) & cspan  # processing segment
                subcache = cache.sieve(segment=pseg)
            data = io_read(cls, subcache, channel, start=pseg[0], end=pseg[1],
                           **kwargs)
            if resample:
                data = data.resample(resample)
            if overlap:  # crop out the overlap introduced for resampling
                return data.crop(*segment)
            return data
        except Exception as exc:  # pylint: disable=broad-except
            if nproc == 1:  # raise now
                raise
            return exc  # will be raised after multiprocessing closes

    # multi-process
    chunks = multiprocess_with_queues(nproc, _read_segment, segs,
                                      raise_exceptions=True, verbose=verbose)

    # flatten dicts
    if issubclass(cls, dict):
        out = cls()
        while chunks:
            new = chunks.pop(0)
            out.append(new, gap=gap, pad=pad)
            del new
        return out

    # flatten series
    list_ = TimeSeriesBaseList(*chunks)
    return list_.join(pad=pad, gap=gap)


def get_mp_cache_segments(cache, nproc, span, overlap=0):
    """Determine segments in which to multiprocess a cache
    """
    # no data
    if not cache:
        nproc = 1
        return [(None, None)]

    # single process
    if nproc == 1:
        return SegmentList([span])

    # actual multiprocessing
    numf = len(cache)
    nproc = min(nproc, numf)
    fperproc = int(ceil(numf / nproc))
    segs = SegmentList()
    # loop over data segments
    for seg in cache_segments(cache) & SegmentList([span]):
        subcache = cache.sieve(segment=(seg.protract(overlap) & span))
        numf2 = len(subcache)
        # if seg is small, process in one
        if numf2 <= fperproc:
            segs.append(seg)
        # otherwise split into chunks
        else:
            j = numf2 / fperproc  # nproc to use here
            dur = ceil(abs(seg) / j)  # time to include in proc
            start, end = seg
            while start + dur <= end:
                segs.append(Segment(start, start + dur))
                start += dur
    return segs
