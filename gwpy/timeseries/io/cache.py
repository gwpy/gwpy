# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""I/O utilities for reading `TimeSeries` from a `list` of file paths.
"""

from io import BytesIO
from math import inf
from os import PathLike

from ...io.cache import (
    FILE_LIKE,
    file_segment,
    read_cache,
    write_cache,
)
from ...segments import Segment

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def preformat_cache(cache, start=None, end=None, sort=file_segment):
    """Preprocess a `list` of file paths for reading.

    This function does the following:

    - read the list of paths cache file (if necessary),
    - sort the cache in time order (if possible),
    - sieve the cache to only include data we need.

    Parameters
    ----------
    cache : `list`, `str`, `pathlib.Path`
        List of file paths, or path to a cache file.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    sort : `callable`, optional
        A callable key function by which to sort the file paths.

    Returns
    -------
    modcache : `list`
        A parsed, sieved list of paths based on the input arguments.

    See also
    --------
    gwpy.io.cache.read_cache
        For details of how the sorting and sieving is implemented
    """
    # if given a list of paths, write it to a file-like structure
    # so that we can use read_cache to do all the work
    if not isinstance(cache, (str, PathLike) + FILE_LIKE):
        cachef = BytesIO()
        write_cache(cache, cachef)
        cachef.seek(0)
        cache = cachef

    # need start and end times to sieve the cache
    if start is None:
        start = -inf
    if end is None:
        end = +inf

    # read the cache
    return read_cache(
        cache,
        coltype=type(start),
        sort=sort,
        segment=Segment(start, end),
    )
