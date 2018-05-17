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
import tempfile
import warnings
from collections import OrderedDict
from gzip import GzipFile

from six import string_types
from six.moves import StringIO

try:
    from lal.utils import CacheEntry
except ImportError:
    HAS_CACHEENTRY = False
else:
    HAS_CACHEENTRY = True

try:
    from glue.lal import Cache
except ImportError:
    HAS_CACHE = False
else:
    HAS_CACHE = True

from ..time import LIGOTimeGPS

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# build list of file-like types
try:  # python2.x
    FILE_LIKE = (
        file, GzipFile,
        tempfile._TemporaryFileWrapper,  # pylint: disable=protected-access
    )
except NameError:  # python3.x
    from io import IOBase
    FILE_LIKE = (
        IOBase, GzipFile,
        tempfile._TemporaryFileWrapper,  # pylint: disable=protected-access
    )


# -- cache I/O ----------------------------------------------------------------

def read_cache(lcf, coltype=LIGOTimeGPS):
    """Read a LAL-format cache file

    Parameters
    ----------
    lcf : `str`, `file`
        Input file or file path to read

    coltype : `LIGOTimeGPS`, `int`, optional
        Type for GPS times

    Returns
    -------
    cache : `list` of :class:`lal.utils.CacheEntry`

    Notes
    -----
    This method requires |lal|_.
    """
    from lal.utils import CacheEntry  # pylint: disable=redefined-outer-name

    # open file
    if not isinstance(lcf, FILE_LIKE):
        with open(lcf, 'r') as fobj:
            return read_cache(fobj, coltype=coltype)

    # read file
    out = []
    append = out.append
    for line in lcf:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        append(CacheEntry(line, coltype=coltype))
    return out


def open_cache(*args, **kwargs):  # pragma: no cover
    # pylint: disable=missing-docstring
    warnings.warn("gwpy.io.cache.open_cache was renamed read_cache",
                  DeprecationWarning)
    return read_cache(*args, **kwargs)


def write_cache(cache, fobj):
    """Write a `list` of cache entries to a file

    Parameters
    ----------
    cache : `list` of :class:`lal.utils.CacheEntry`
        The cache to write

    fobj : `file`, `str`
        The open file object, or file path to write to.
    """
    # open file
    if isinstance(fobj, string_types):
        with open(fobj, 'w') as fobj2:
            return write_cache(cache, fobj2)

    # write file
    for entry in cache:
        line = '{0}\n'.format(entry)
        try:
            fobj.write(line)
        except TypeError:  # python3 'wb' mode
            fobj.write(line.encode('utf-8'))


def is_cache(cache):
    """Returns `True` if ``cache`` is a readable cache file or object

    Parameters
    ----------
    cache : `str`, `file`, `list`
        Object to detect as cache

    Returns
    -------
    iscache : `bool`
        `True` if the input object is a cache, or a file in LAL cache format,
        otherwise `False`
    """
    if isinstance(cache, string_types + FILE_LIKE):
        try:
            return bool(len(read_cache(cache)))
        except (TypeError, ValueError, UnicodeDecodeError, ImportError):
            # failed to parse cache
            return False
    if HAS_CACHE and isinstance(cache, Cache):
        return True
    if (isinstance(cache, (list, tuple)) and cache and
            all(map(is_cache_entry, cache))):
        return True

    return False


def is_cache_entry(path):
    """Returns `True` if ``path`` can be represented as a cache entry

    In practice this just tests whether the input is |LIGO-T050017|_ compliant.

    Parameters
    ----------
    path : `str`, :class:`lal.utils.CacheEntry`
        The input to test

    Returns
    -------
    isentry : `bool`
        `True` if ``path`` is an instance of `CacheEntry`, or can be parsed
        using |LIGO-T050017|_.
    """
    if HAS_CACHEENTRY and isinstance(path, CacheEntry):
        return True
    try:
        file_segment(path)
    except (ValueError, TypeError, AttributeError):
        return False
    return True


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
        - :class:`~lal.utils.CacheEntry`
        - `str` with ``.cache`` or ``.lcf`` extension
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
        return [e.path for e in read_cache(flist)]

    # separate comma-separate list of names
    if isinstance(flist, string_types):
        return flist.split(',')

    # parse list of entries (of some format)
    if isinstance(flist, (list, tuple)):
        return list(map(file_name, flist))

    # otherwise parse a single entry
    try:
        return [file_name(flist)]
    except ValueError as exc:
        exc.args = ("Could not parse input %r as one or more "
                    "file-like objects" % flist,)
        raise


def file_name(fobj):
    """Returns the name (path) of the file object
    """
    if isinstance(fobj, string_types):
        return fobj
    if isinstance(fobj, FILE_LIKE) and not isinstance(fobj, StringIO):
        return fobj.name
    if HAS_CACHEENTRY and isinstance(fobj, CacheEntry):
        return fobj.path
    raise ValueError("Cannot parse file name for %r" % fobj)


def file_segment(filename):
    """Return the data segment for a filename following T050017

    Parameters
    ---------
    filename : `str`, :class:`~lal.utils.CacheEntry`
        the path name of a file

    Returns
    -------
    segment : `~gwpy.segments.Segment`
        the ``[start, stop)`` GPS segment covered by the given file

    Notes
    -----
    |LIGO-T050017|_ declares a filenaming convention that includes
    documenting the GPS start integer and integer duration of a file,
    see that document for more details.
    """
    from ..segments import Segment
    try:  # CacheEntry
        return Segment(filename.segment)
    except AttributeError:  # file path (str)
        base = os.path.basename(filename)
        try:
            _, _, start, end = base.split('-')
        except ValueError as exc:
            exc.args = ('Failed to parse {0!r} as LIGO-T050017-compatible '
                        'filename'.format(base),)
            raise
        start = float(start)
        end = float(end.split('.')[0])
        return Segment(start, start+end)


def cache_segments(*caches):
    """Returns the segments of data covered by entries in the cache(s).

    Parameters
    ----------
    *caches : `list`
        One or more lists of file paths
        (`str` or :class:`~lal.utils.CacheEntry`).

    Returns
    -------
    segments : `~gwpy.segments.SegmentList`
        A list of segments for when data should be available
    """
    from ..segments import SegmentList
    out = SegmentList()
    for cache in caches:
        out.extend(file_segment(e) for e in cache)
    return out.coalesce()


def flatten(*caches):
    """Flatten a nested list of cache entries

    Parameters
    ----------
    *caches : `list`
        One or more lists of file paths
        (`str` or :class:`~lal.utils.CacheEntry`).

    Returns
    -------
    flat : `list`
        A flat `list` containing the unique set of entries across
        each input.
    """
    return list(OrderedDict.fromkeys(e for c in caches for e in c))


def find_contiguous(*caches):
    """Separate one or more cache entry lists into time-contiguous sub-lists

    Parameters
    ----------
    *caches : `list`
        One or more lists of file paths
        (`str` or :class:`~lal.utils.CacheEntry`).

    Returns
    -------
    caches : `iter` of `list`
        an interable yielding each contiguous cache
    """
    flat = flatten(*caches)
    for segment in cache_segments(flat):
        yield sieve(flat, segment=segment)


def sieve(cache, segment=None):
    """Filter the cache to find those entries that overlap ``segment``

    Parameters
    ----------
    cache : `list`
        Input list of file paths

    segment : `~gwpy.segments.Segment`
        The ``[start, stop)`` interval to match against.
    """
    return type(cache)(e for e in cache if segment.intersects(file_segment(e)))
