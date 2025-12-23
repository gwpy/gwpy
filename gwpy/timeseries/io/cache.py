# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""I/O utilities for reading `TimeSeries` from a `list` of file paths."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from igwn_segments import infinity

from ...io.cache import (
    file_segment,
    read_cache,
    write_cache,
)
from ...io.utils import Readable
from ...segments import Segment

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )

    from ...io.utils import FileSystemPath
    from ...time import SupportsToGps

inf = infinity()

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def preformat_cache(
    cache: Readable | Iterable[FileSystemPath],
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    sort: Callable = file_segment,
) -> list[str]:
    """Preprocess a `list` of file paths for reading.

    This function does the following:

    1. read the list of paths cache file (if necessary),
    2. sort the cache in time order (if possible),
    3. sieve the cache to only include data we need.

    Parameters
    ----------
    cache : `list`, `str`, `pathlib.Path`, `file`
        List of file paths, or a reference to a cache file.

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

    See Also
    --------
    gwpy.io.cache.read_cache
        For details of how the sorting and sieving is implemented
    """
    # if given a list of paths, write it to a file-like structure
    # so that we can use read_cache to do all the work
    if not isinstance(cache, Readable):
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
