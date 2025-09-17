# Copyright (c) 2023-2025 Cardiff University
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

"""Tests for :mod:`gwpy.timeseries.io.cache`."""

import pytest

from ...io.cache import write_cache
from ..io import cache as ts_io_cache


@pytest.fixture
def cache():
    """List of files over which to test sorting/sieving."""
    return [
        "/tmp/A-TEST-0-10.tmp",
        "/tmp/A-TEST-10-10.tmp",
        "/tmp/A-TEST-20-10.tmp",
        "/tmp/A-TEST-30-5.tmp",
        "/tmp/A-TEST-35-15.tmp",
    ]


@pytest.fixture
def cache_file(tmp_path, cache):
    """File version of `cache()`."""
    path = tmp_path / "cache.txt"
    write_cache(cache, path)
    return path


@pytest.mark.parametrize("source", ("cache", "cache_file"))
@pytest.mark.parametrize(("start", "end", "idx"), [
    # use everything in the cache
    (None, None, slice(None)),
    # use only GPS time '25' onwards, which is cache[2:]
    (25, None, slice(2, None)),
    # use only up to GPS time '25', which is cache[:3]
    (None, 25, slice(None, 3)),
    # use interval [10, 35), which needs cache[1:4]
    (10, 35, slice(1, 4)),
])
def test_preformat_cache(request, cache, source, start, end, idx):
    """Test that `gwpy.timeseries.io.cache.preformat_cache` works properly.

    Here `[start, end)` is a GPS segment, and `idx` the corresponding slice
    needed to restrict the cache object.

    Loops over a variety of input arguments, using `request` to dynamically
    loop over `cache` or `cache_file` as the input.
    """
    assert ts_io_cache.preformat_cache(
        request.getfixturevalue(source),  # cache or cache_file
        start=start,
        end=end,
    ) == cache[idx]
