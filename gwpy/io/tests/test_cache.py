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

"""Unit test for `io` module."""

import tempfile
from copy import deepcopy
from pathlib import Path

import numpy
import pytest

from ...segments import Segment, SegmentList
from .. import cache as io_cache

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

SEGMENTS = SegmentList(map(Segment, [
    (0, 1),
    (1, 2),
    (4, 5),
]))
CACHE = [Path("tmp") / f"A-B-{seg[0]}-{seg[1]-seg[0]}.tmp" for seg in SEGMENTS]


# -- fixtures -----------------------------------------------------------------

@pytest.fixture
def cache():
    """Return a sample cache."""
    return list(map(str, CACHE))


@pytest.fixture
def segments():
    """Return a sample SegmentList."""
    return deepcopy(SEGMENTS)


# -- tests --------------------------------------------------------------------

@pytest.mark.parametrize(("fmt", "entry1"), [
    (None, str(CACHE[0])),
    pytest.param(
        "lal",
        f"A B {SEGMENTS[0][0]} {abs(SEGMENTS[0])} {CACHE[0]}",
        id="lal"),
    pytest.param(
        "ffl",
        f"{CACHE[0]} {SEGMENTS[0][0]} {abs(SEGMENTS[0])} 0 0",
        id="ffl"),
])
def test_read_write_cache(cache, tmp_path, fmt, entry1):
    """Test `read_cache()` and `write_cache()`."""
    tmp = tmp_path / "test.lcf"

    # write cache using filename
    io_cache.write_cache(cache, tmp, format=fmt)

    # check that first line is proper LAL format
    with tmp.open("r") as tmpf:
        assert tmpf.readline().strip() == entry1

    # now read it back ans check we get the same answer
    assert io_cache.read_cache(tmp) == cache

    # check read/write with a file object
    with tmp.open("w") as tmpf:
        io_cache.write_cache(cache, tmpf, format=fmt)
    with tmp.open("r") as tmpf:
        assert io_cache.read_cache(tmpf) == cache

    # check sieving and sorting works
    assert io_cache.read_cache(
        tmp,
        sort=lambda e: -io_cache.filename_metadata(e)[2][0],
        segment=Segment(0, 2),
    ) == cache[1::-1]


@pytest.mark.requires("lal.utils")
def test_write_cache_cacheentry(cache, tmp_path):
    """Test `write_cache()` with `lal.CacheEntry` objects."""
    from lal.utils import CacheEntry
    tmp = tmp_path / "test.lcf"
    lcache = list(map(CacheEntry.from_T050017, cache))
    with tmp.open("w") as tmpf:
        io_cache.write_cache(lcache, tmpf, format=None)

    # check first line looks like a LAL-format cache entry
    with tmp.open("r") as tmpf:
        assert tmpf.readline().strip() == (
            f"A B {SEGMENTS[0][0]} {abs(SEGMENTS[0])} {CACHE[0]}"
        )

    # read from file name
    assert io_cache.read_cache(tmp) == cache


@pytest.mark.parametrize(("input_", "result"), [
    (None, False),
    ([], False),
    (["A-B-12345-6.txt"], True),
])
def test_is_cache(input_, result):
    """Test `is_cache()`."""
    assert io_cache.is_cache(input_) is result


@pytest.mark.requires("lal.utils")
def test_is_cache_lal():
    """Test `is_cache()` with `lal.CacheEntry` objects."""
    cache = [io_cache.CacheEntry.from_T050017("/data/A-B-12345-6.txt")]
    assert io_cache.is_cache(cache)
    assert not io_cache.is_cache([*cache, None])  # type: ignore[list-item]


@pytest.mark.requires("glue.lal")
def test_is_cache_glue():
    """Test `is_cache()` with `glue.lal.Cache` objects."""
    assert io_cache.is_cache(io_cache.Cache())

    # check ASCII file gets returned as False
    a = numpy.array([[1, 2], [3, 4]])
    with tempfile.TemporaryFile() as f:
        numpy.savetxt(f, a)
        f.seek(0)
        assert io_cache.is_cache(f) is False


@pytest.mark.requires("lal.utils")
def test_is_cache_file(tmp_path):
    """Check that `gwpy.io.cache.is_cache` returns `True` when it should."""
    # write a cache file
    e = io_cache.CacheEntry.from_T050017("/data/A-B-12345-6.txt")
    tmp = tmp_path / "tmpfile"
    io_cache.write_cache([e], tmp)

    # check that we can identify it properly
    assert io_cache.is_cache(tmp)  # file name
    with tmp.open("r") as tmpf:
        assert io_cache.is_cache(tmpf)  # open file object


@pytest.mark.requires("lal.utils")
def test_is_cache_file_empty(tmp_path):
    """Check that `gwpy.io.cache.is_cache` returns False when it should."""
    tmp = tmp_path / "tmpfile"
    assert not io_cache.is_cache(tmp)  # FileNotFoundError
    tmp.touch()
    assert not io_cache.is_cache(tmp)  # empty file


def test_is_cache_entry():
    """Test `is_cache_entry()`."""
    assert io_cache.is_cache_entry("/data/A-B-12345-6.txt")
    assert not io_cache.is_cache_entry("random-file-name.blah")
    try:
        e = io_cache.CacheEntry.from_T050017("/data/A-B-12345-6.txt")
    except AttributeError:
        pass
    else:
        assert io_cache.is_cache_entry(e)


def test_cache_segments(cache, segments):
    """Test `cache_segments()`."""
    # check empty input
    sl = io_cache.cache_segments()
    assert isinstance(sl, SegmentList)
    assert len(sl) == 0

    # check simple cache
    segments.coalesce()
    sl = io_cache.cache_segments(cache)
    assert sl == segments

    # check multiple caches produces the same result
    sl = io_cache.cache_segments(cache[:2], cache[2:])
    assert sl == segments


@pytest.mark.parametrize(("path", "metadata"), [
    pytest.param(
        "A-B-0-1.txt",
        ("A", "B", Segment(0, 1)),
        id="int",
    ),
    pytest.param(
        "/path/to/A-B-0.456-1.345.txt.gz",
        ("A", "B", Segment(0.456, 1.801)),
        id="float",
    ),
])
def test_filename_metadata(path, metadata):
    """Test `filename_metadata`."""
    assert io_cache.filename_metadata(path) == metadata


def test_filename_metadata_error():
    """Test `filename_metadata` error handling."""
    with pytest.raises(
        ValueError,
        match=r"^could not convert string to float",
    ):
        io_cache.filename_metadata("A-B-0-4xml.gz")


@pytest.mark.parametrize(("filename", "seg"), [
    # basic
    ("A-B-1-2.ext", Segment(1, 3)),
    # multiple file extensions
    ("A-B-1-2.ext.gz", Segment(1, 3)),
    # non-integer
    ("A-B-1.23-4.ext.gz", Segment(1.23, 5.23)),
])
def test_file_segment(filename, seg):
    """Test `file_segment()`."""
    fs = io_cache.file_segment(filename)
    assert isinstance(fs, Segment)
    assert fs == seg

    # check mutliple file extensions
    assert io_cache.file_segment("A-B-1-2.ext.gz") == (1, 3)

    # check floats (and multiple file extensions)
    assert io_cache.file_segment("A-B-1.23-4.ext.gz") == (1.23, 5.23)


def test_file_segment_errors():
    """Test :func:`gwpy.io.cache.file_segment` error handling."""
    with pytest.raises(
        ValueError,
        match=r"^Failed to parse 'blah' as a LIGO-T010150-compatible filename$",
    ):
        io_cache.file_segment("blah")


def test_flatten(cache):
    """Test `flatten()`."""
    # check flattened version of single cache is unchanged
    assert io_cache.flatten(cache) == cache
    assert io_cache.flatten(cache, cache) == cache

    # check two caches get concatenated properly
    a = cache
    b = [e.replace("A-B-", "A-B-1") for e in cache]
    c = a + b
    assert io_cache.flatten(a, b) == c


def test_find_contiguous(cache, segments):
    """Test `find_contiguous()`."""
    segments.coalesce()
    for i, subcache in enumerate(io_cache.find_contiguous(cache)):
        assert io_cache.cache_segments(subcache).extent() == segments[i]

    assert not list(io_cache.find_contiguous())


def test_sieve(cache, segments):
    """Test `sieve()`."""
    sieved = io_cache.sieve(cache, segments[0])
    assert type(sieved) is type(cache)
    assert sieved == cache[:1]

    segments.coalesce()
    assert io_cache.sieve(cache, segments[0]) == cache[:2]


def test_sieve_strict():
    """Test `sieve(..., strict={True,False})`."""
    cache = [
        "A-B-0-1.txt",
        "A-B-1-1.txt",
        "A-B-2-1.txt",
        "somethingelse.txt",
    ]

    # check that strict=True raises an error
    with pytest.raises(
        ValueError,
        match=r"^Failed to parse",
    ):
        io_cache.sieve(cache, Segment(0, 2), strict=True)

    # but strict=False only emits a warning
    with pytest.warns(
        UserWarning,
        match=r"^Failed to parse",
    ):
        assert (
            io_cache.sieve(cache, Segment(0, 2), strict=False)
            == cache[:2]
        )
