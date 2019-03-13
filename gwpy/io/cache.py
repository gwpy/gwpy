# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

"""Input/Output utilities for GW Cache files.

A cache file is a specially-formatted ASCII file that contains file paths
and associated metadata for those files, designed to make identifying
relevant data, and sieving large file lists, easier for the user.
"""

from __future__ import division

import os.path
import warnings
from collections import (namedtuple, OrderedDict)

from six import string_types

from ..time import LIGOTimeGPS
from .utils import (FILE_LIKE, file_path)

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

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class _CacheEntry(namedtuple(
        '_CacheEntry', ['observatory', 'description', 'segment', 'path'])):
    """Quick version of lal.utils.CacheEntry for internal purposes only

    Just to allow metadata handling for files that don't follow LIGO-T050017.
    """
    def __str__(self):
        return self.path

    @classmethod
    def parse(cls, line, gpstype=LIGOTimeGPS):
        from ..segments import Segment

        if isinstance(line, bytes):
            line = line.decode('utf-8')

        # determine format
        try:
            # NOTE: cast to `str` here to avoid unicode on python2.7
            #       which lal.LIGOTimeGPS doesn't like
            a, b, c, d, e = map(str, line.strip().split())
        except ValueError as exc:
            exc.args = ("Cannot identify format for cache entry {!r}".format(
                line.rstrip()),)
            raise

        try:  # Virgo FFL format includes GPS start time in second place
            start = gpstype(b)
        except (RuntimeError, TypeError, ValueError):  # LAL format
            start = gpstype(c)
            end = start + float(d)
            return cls(a, b, Segment(start, end), e)
        path = a
        start = float(b)
        end = start + float(c)
        try:
            observatory, description = os.path.splitext(
                os.path.basename(path))[0].split('-', 2)[:2]
        except ValueError:
            return cls(None, None, Segment(start, end), path)
        return cls(observatory, description, Segment(start, end), path)


# -- cache I/O ----------------------------------------------------------------

def _iter_cache(cachefile, gpstype=LIGOTimeGPS):
    """Internal method that yields a `_CacheEntry` for each line in the file

    This method supports reading LAL- and (nested) FFL-format cache files.
    """
    try:
        path = os.path.abspath(cachefile.name)
    except AttributeError:
        path = None
    for line in cachefile:
        try:
            yield _CacheEntry.parse(line, gpstype=LIGOTimeGPS)
        except ValueError:
            # virgo FFL format (seemingly) supports nested FFL files
            parts = line.split()
            if len(parts) == 3 and os.path.abspath(parts[0]) != path:
                with open(parts[0], 'r') as cache2:
                    for entry in _iter_cache(cache2):
                        yield entry
            else:
                raise


def read_cache(cachefile, coltype=LIGOTimeGPS, sort=None, segment=None):
    """Read a LAL- or FFL-format cache file as a list of file paths

    Parameters
    ----------
    cachefile : `str`, `file`
        Input file or file path to read.

    coltype : `LIGOTimeGPS`, `int`, optional
        Type for GPS times.

    sort : `callable`, optional
        A callable key function by which to sort the output list of file paths

    segment : `gwpy.segments.Segment`, optional
        A GPS `[start, stop)` interval, if given only files overlapping this
        interval will be returned.

    Returns
    -------
    paths : `list` of `str`
        A list of file paths as read from the cache file.

    Notes
    -----
    This method requires |lal|_.
    """
    # open file
    if not isinstance(cachefile, FILE_LIKE):
        with open(file_path(cachefile), 'r') as fobj:
            return read_cache(fobj, coltype=coltype, sort=sort,
                              segment=segment)

    # read file
    cache = list(_iter_cache(cachefile, gpstype=coltype))

    # sort and sieve
    if sort:
        cache.sort(key=sort)
    if segment:
        cache = sieve(cache, segment=segment)

    # read simple paths
    return [x.path for x in cache]


def read_cache_entry(line):
    """Read a file path from a line in a cache file.

    Parameters
    ----------
    line : `str`, `bytes`
        Line of text to parse

    Returns
    -------
    path : `str`
       The file path.

    Raises
    ------
    ValueError
        if the line cannot be parsed successfully
    """
    return _CacheEntry.parse(line).path


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


# -- moved functions ----------------------------------------------------------

def file_name(*args, **kwargs):
    from .utils import file_path
    warnings.warn("this function has been moved to gwpy.io.utils.file_path",
                  DeprecationWarning)
    return file_path(*args, **kwargs)


def file_list(*args, **kwargs):
    from .utils import file_list
    warnings.warn("this function has been moved to gwpy.io.utils.file_list",
                  DeprecationWarning)
    return file_list(*args, **kwargs)
