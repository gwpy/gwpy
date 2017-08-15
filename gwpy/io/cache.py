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

"""Input/Output utilities for LAL Cache files.
"""

from __future__ import division

import os.path
from math import ceil
from multiprocessing import (cpu_count, Process, Queue as ProcessQueue)
from six import string_types
import tempfile
import warnings
from gzip import GzipFile

from six import string_types

from numpy import recarray
from numpy.lib import recfunctions

try:
    from glue.lal import Cache
except ImportError:  # no lal
    HAS_CACHE = False
else:
    HAS_CACHE = True
    from lal.utils import CacheEntry
    Cache.entry_class = CacheEntry

from astropy.table import (Table, vstack as vstack_tables)
from astropy.io.registry import _get_valid_format

from ..time import LIGOTimeGPS

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# build list of file-like types
try:  # python2.x
    FILE_LIKE = [file, GzipFile]
except NameError:  # python3.x
    from io import IOBase
    FILE_LIKE = [IOBase, GzipFile]
try:  # protect against private member being removed
    FILE_LIKE.append(tempfile._TemporaryFileWrapper)
except AttributeError:
    pass
FILE_LIKE = tuple(FILE_LIKE)


# -- cache I/O ----------------------------------------------------------------

def read_cache(lcf, coltype=LIGOTimeGPS):
    """Read a LAL-format cache file into memory as a `Cache`

    Parameters
    ----------
    lcf : `str`, `file`
        input file or file path to read

    coltype : `LIGOTimeGPS`, `int`, optional
        `type` for GPS times

    Returns
    -------
    cache : :class:`glue.lal.Cache`
        a cache object, representing each line in the file as a
        :class:`~lal.utils.CacheEntry`
    """
    from glue.lal import Cache
    # open file
    if not isinstance(lcf, FILE_LIKE):
        with open(lcf, 'r') as f:
            return open_cache(f, coltype=coltype)

    # read file
    out = Cache()
    for line in lcf:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        out.append(out.entry_class(line, coltype=coltype))
    return out


def open_cache(*args, **kwargs):
    warnings.warn("gwpy.io.cache.open_cache was renamed read_cache",
                  DeprecationWarning)
    return read_cache(*args, **kwargs)


def write_cache(cache, fobj):
    """Write a :cache:`~glue.lal.Cache` to a file

    Parameters
    ----------
    cache : :class:`glue.lal.Cache`
        the cache to write

    fobj : `file`, `str`
        the open file object, or file path to write to
    """
    # open file
    if isinstance(fobj, string_types):
        with open(fobj, 'w') as f:
            return write_cache(cache, f)

    # write file
    for entry in cache:
        line = '%s\n' % entry
        try:
            fobj.write(line)
        except TypeError:
            fobj.write(line.encode('utf-8'))


# -- cache manipulation -------------------------------------------------------

def file_list(flist):
    """Parse a number of possible input types into a list of filepaths.

    Parameters
    ----------
    flist : `file-like` or `list-like` iterable
        the input data container, normally just a single file path, or a list
        of paths, but can generally be any of the following

        - `str` representing a single file path (or comma-separated collection)
        - open `file` or `~gzip.GzipFile` object
        - `~lal.utils.CacheEntry`
        - `~glue.lal.Cache` object or `str` with '.cache' or '.lcf extension
        - simple `list` or `tuple` of `str` paths

    Returns
    -------
    files : `list`
        `list` of `str` file paths

    Raises
    ------
    ValueError
        if the input `flist` cannot be interpreted as any of the above inputs
    """
    # open a cache file and return list of paths
    if isinstance(flist, string_types) and flist.endswith(('.cache', '.lcf')):
        return read_cache(flist).pfnlist()

    # separate comma-separate list of names
    if isinstance(flist, string_types):
        return flist.split(',')

    # parse list of entries (of some format)
    if isinstance(flist, (list, tuple)):
        return list(map(file_name, flist))

    # otherwise parse a single entry
    try:
        return [file_name(flist)]
    except ValueError as e:
        e.args = ("Could not parse input %r as one or more "
                  "file-like objects" % flist,)
        raise


def file_name(fobj):
    """Returns the name (path) of the file object
    """
    if isinstance(fobj, string_types):
        return fobj
    if isinstance(fobj, FILE_LIKE):
        return fobj.name
    if HAS_CACHE and isinstance(fobj, CacheEntry):
        return fobj.path
    raise ValueError("Cannot parse file name for %r" % fobj)


def file_segment(filename):
    """Return the data segment for a filename following T050017

    Parameters
    ---------
    filename : `str`, `~lal.utils.CacheEntry`
        the path name of a file

    Returns
    -------
    segment : `~gwpy.segments.Segment`
        the ``[start, stop)`` GPS segment covered by the given file

    Notes
    -----
    `LIGO-T050017 <https://dcc.ligo.org/LIGO-T050017/public>`_ declares
    a filenaming convention that includes documenting the GPS start integer
    and integer duration of a file, see that document for more details.
    """
    try:  # filename object provides its own segment information
        return filename.segment
    except AttributeError:  # otherwise parse from T050017 spec
        from ..segments import Segment
        base = os.path.basename(filename)
        try:
            _, _, s, e = base.split('-')
        except ValueError as e:
            e.args = ('Failed to parse %r as LIGO-T050017-compatible filename'
                      % filename,)
            raise
        s = float(s)
        e = int(e.split('.')[0])
        return Segment(s, s+e)


def cache_segments(*caches):
    """Build a `SegmentList` of data availability for these `Caches`

    Parameters
    ----------
    *cache : `~glue.lal.Cache`, `list`
        one of more `Cache` objects (or simple `list`)

    Returns
    -------
    segments : `~gwpy.segments.SegmentList`
        a list of segments for when data should be available
    """
    from ..segments import SegmentList
    out = SegmentList()
    for cache in caches:
        out.extend(file_segment(e) for e in cache)
    return out.coalesce()


def flatten(*caches):
    """Flatten a list of :class:`Caches <glue.lal.Cache>` into a single cache

    Parameters
    ----------
    *caches
        one or more :class:`~glue.lal.Cache` objects

    Returns
    -------
    flat : :class:`~glue.lal.Cache`
        a single cache containing the unique set of entries across
        each input
    """
    cache_type = type(caches[0])
    return cache_type([e for c in caches for e in c]).unique()


def find_contiguous(*caches):
    """Separate one or more caches into sets of contiguous caches

    Parameters
    ----------
    *caches
        one or more :class:`~glue.lal.Cache` objects

    Returns
    -------
    caches : `iter` of :class:`~glue.lal.Cache`
        an interable yielding each contiguous cache
    """
    from glue.lal import Cache
    try:
        flat = flatten(*caches)
    except IndexError:
        flat = Cache()
    for segment in cache_segments(flat):
        yield flat.sieve(segment=segment)
