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

from ...io.cache import (FILE_LIKE, read_cache, file_segment, sieve)
from ...segments import Segment

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def preformat_cache(cache, start=None, end=None):
    """Preprocess a `list` of file paths for reading.

    - read the cache from the file (if necessary)
    - sieve the cache to only include data we need

    Parameters
    ----------
    cache : `list`, `str`
        List of file paths, or path to a LAL-format cache file on disk.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    Returns
    -------
    modcache : `list`
        A parsed, sieved list of paths based on the input arguments.
    """
    # open cache file
    if isinstance(cache, (str,) + FILE_LIKE):
        return read_cache(cache, sort=file_segment,
                          segment=Segment(start, end))

    # format existing cache file
    cache = type(cache)(cache)  # copy cache

    # sort cache
    try:
        cache.sort(key=file_segment)  # sort
    except ValueError:
        # if this failed, then the sieving will also fail, but lets proceed
        # anyway, since the user didn't actually ask us to do this (but
        # its a very good idea)
        return cache

    # sieve cache
    if start is None:  # start time of earliest file
        start = file_segment(cache[0])[0]
    if end is None:  # end time of latest file
        end = file_segment(cache[-1])[-1]
    return sieve(cache, segment=Segment(start, end))
