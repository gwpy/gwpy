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

"""Unit tests for :mod:`gwpy.io.utils`."""

import gzip
import tempfile
from pathlib import Path

import pytest

from .. import (
    cache as io_cache,
    utils as io_utils,
)
from .test_cache import cache  # noqa: F401

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def test_gopen(tmp_path):
    """Test `gopen` with a normal file."""
    tmp = tmp_path / "test.tmp"
    # test simple use
    with tmp.open("w") as f:
        f.write("blah blah blah")
    with pytest.deprecated_call():  # noqa: SIM117
        with io_utils.gopen(tmp) as f2:
            assert f2.read() == "blah blah blah"


@pytest.mark.parametrize("suffix", [".txt.gz", ""])
def test_gopen_gzip(tmp_path, suffix):
    """Test `gopen` with a gzip file."""
    tmp = tmp_path / f"test{suffix}"
    text = b"blah blah blah"
    with gzip.open(tmp, "wb") as fobj:
        fobj.write(text)
    with (
        pytest.deprecated_call(),
        io_utils.gopen(tmp, mode="rb") as fobj2,
    ):
        assert isinstance(fobj2, gzip.GzipFile)
        assert fobj2.read() == text


def test_file_list_file():
    """Test `file_list` with different file-like objects."""
    # test file -> [file.name]
    with tempfile.NamedTemporaryFile() as f:
        assert io_utils.file_list(f) == [f.name]


def test_file_list_cache(cache):  # noqa: F811
    """Test `file_list` with `lal.CacheEntry` objects."""
    lal_utils = pytest.importorskip("lal.utils")
    # test CacheEntry -> [CacheEntry.path]
    lcache = list(map(lal_utils.CacheEntry.from_T050017, cache))
    assert io_utils.file_list(lcache[0]) == [cache[0]]

    # test cache object -> pfnlist
    assert io_utils.file_list(lcache) == cache

    # test cache file -> pfnlist()
    with tempfile.NamedTemporaryFile(suffix=".lcf", mode="w") as f:
        io_cache.write_cache(lcache, f)
        f.seek(0)
        assert io_utils.file_list(f.name) == cache


@pytest.mark.parametrize("arg", [
    pytest.param("A,B,C,D", id="comma-separated"),
    pytest.param(["A", "B", "C", "D"], id="list"),
])
def test_file_list_str(arg):
    """Test `file_list` with different string inputs."""
    assert io_utils.file_list(arg) == ["A", "B", "C", "D"]


def test_file_list_error():
    """Test `file_list` with a bad input."""
    with pytest.raises(
        ValueError,
        match=r"^Could not parse input 1 as one or more file-like objects$",
    ):
        io_utils.file_list(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(("input_", "expected"), [
    ("test.txt", "test.txt"),
    ("file:///test/path.txt", "/test/path.txt"),
    (Path("test.txt"), "test.txt"),
])
def test_file_path(input_, expected):
    """Test `file_path()`."""
    assert io_utils.file_path(input_) == expected


def test_file_path_file():
    """Check that :func:`gwpy.io.utils.file_path` can handle open files."""
    with tempfile.NamedTemporaryFile() as f:
        assert io_utils.file_path(f) == f.name


@pytest.mark.parametrize("badthing", [
    1,
    ["test.txt"],
])
def test_file_path_errors(badthing):
    """Check that :func:`gwpy.io.utils.file_path` fails when expected."""
    with pytest.raises(
        ValueError,
        match=r"^cannot parse file name for ",
    ):
        io_utils.file_path(badthing)


def test_file_path_cacheentry():
    """Check that :func:`gwpy.io.utils.file_path` can handle `CacheEntry`."""
    lal_utils = pytest.importorskip("lal.utils")
    path = "/path/to/A-B-0-1.txt"
    assert io_utils.file_path(lal_utils.CacheEntry.from_T050017(path)) == path
