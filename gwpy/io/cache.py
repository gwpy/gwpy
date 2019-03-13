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

from __future__ import (division, print_function)

import os.path
import warnings
from collections import (namedtuple, OrderedDict)
from pathlib import Path

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


def _preformat_entry(entry):
    path = file_path(entry)
    obs, tag, seg = filename_metadata(path)
    start, end = seg
    if start.is_integer():
        start = int(start)
    if end.is_integer():
        end = int(end)
    return path, obs, tag, type(seg)(start, end)


def _format_entry_lal(entry):
    path, obs, tag, seg = _preformat_entry(entry)
    return "{} {} {} {} {}".format(obs, tag, seg[0], abs(seg), path)


def _parse_entry_lal(line, gpstype=LIGOTimeGPS):
    from ..segments import Segment
    obs, desc, start, dur, path = map(str, line)
    start = gpstype(start)
    end = start + float(dur)
    return _CacheEntry(obs, desc, Segment(start, end), path)


def _format_entry_ffl(entry):
    path, obs, tag, seg = _preformat_entry(entry)
    return "{} {} {} 0 0".format(path, seg[0], abs(seg))


def _parse_entry_ffl(line, gpstype=LIGOTimeGPS):
    from ..segments import Segment
    path, start, dur, _, _ = map(str, line)
    start = gpstype(start)
    end = start + float(dur)
    try:
        observatory, description = Path(path).name.split('-', 2)[:2]
    except ValueError:
        return _CacheEntry(None, None, Segment(start, end), path)
    return _CacheEntry(observatory, description, Segment(start, end), path)


class _CacheEntry(namedtuple(
        '_CacheEntry', ['observatory', 'description', 'segment', 'path'])):
    """Quick version of lal.utils.CacheEntry for internal purposes only

    Just to allow metadata handling for files that don't follow LIGO-T050017.
    """
    def __str__(self):
        return self.path

    @classmethod
    def parse(cls, line, gpstype=LIGOTimeGPS):

        # format line string
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        parts = line.strip().split()

        # if single entry, parse filename
        if len(parts) == 1:
            path = parts[0]
            return cls(*filename_metadata(path) + (path,))

        try:
            return _parse_entry_ffl(parts)
        except (RuntimeError, TypeError, ValueError) as exc:
            try:
                return _parse_entry_lal(parts)
            except ValueError:
                pass
            exc.args = ("Cannot identify format for cache entry {!r}".format(
                line.strip()),)
            raise


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
    """
    # open file
    if not isinstance(cachefile, FILE_LIKE):
        with open(file_path(cachefile), 'r') as fobj:
            return read_cache(fobj, coltype=coltype, sort=sort,
                              segment=segment)

    # read file
    cache = [x.path for x in _iter_cache(cachefile, gpstype=coltype)]

    # sieve and sort
    if segment:
        cache = sieve(cache, segment=segment)
    if sort:
        cache.sort(key=sort)

    # read simple paths
    return cache


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


def write_cache(cache, fobj, format=None):
    """Write a `list` of cache entries to a file

    Parameters
    ----------
    cache : `list` of `str`
        The list of file paths to write

    fobj : `file`, `str`
        The open file object, or file path to write to.

    format : `str`, optional
        The format to write to, one of

        - `None` : format each entry using `str`
        - ``'lal'`` : write a LAL-format cache
        - ``'ffl'`` : write an FFL-format cache
    """
    # open file
    if isinstance(fobj, string_types):
        with open(fobj, 'w') as fobj2:
            return write_cache(cache, fobj2, format=format)

    if format is None:
        formatter = str
    elif format.lower() == "lal":
        formatter = _format_entry_lal
    elif format.lower() == "ffl":
        formatter = _format_entry_ffl
    else:
        raise ValueError("Unrecognised cache format {!r}".format(format))

    # write file
    for line in map(formatter, cache):
        try:
            print(line, file=fobj)
        except TypeError:  # bytes-mode
            fobj.write("{}\n".format(line).encode("utf-8"))


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

def filename_metadata(filename):
    """Return metadata parsed from a filename following LIGO-T050017

    This method is lenient with regards to integers in the GPS start time of
    the file, as opposed to `gwdatafind.utils.filename_metadata`, which is
    strict.

    Parameters
    ----------
    filename : `str`
        the path name of a file

    Returns
    -------
    obs : `str`
        the observatory metadata

    tag : `str`
        the file tag

    segment : `gwpy.segments.Segment`
        the GPS ``[float, float)`` interval for this file

    Notes
    -----
    `LIGO-T050017 <https://dcc.ligo.org/LIGO-T050017>`__ declares a
    file naming convention that includes documenting the GPS start integer
    and integer duration of a file, see that document for more details.

    Examples
    --------
    >>> from gwpy.io.cache import filename_metadata
    >>> filename_metadata("A-B-0-1.txt")
    ('A', 'B', Segment(0, 1))
    >>> filename_metadata("A-B-0.456-1.345.txt")
    ("A", "B", Segment(0.456, 1.801))
    """
    from ..segments import Segment
    name = Path(filename).name
    try:
        obs, desc, start, dur = name.split('-')
    except ValueError as exc:
        exc.args = ('Failed to parse {!r} as LIGO-T050017-compatible '
                    'filename'.format(name),)
        raise
    start = float(start)
    dur = dur.rsplit('.', 1)[0]
    while True:  # recursively remove extension components
        try:
            dur = float(dur)
        except ValueError:
            if '.' not in dur:
                raise
            dur = dur.rsplit('.', 1)[0]
        else:
            break
    return obs, desc, Segment(start, start+dur)


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
        return filename_metadata(filename)[2]


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
