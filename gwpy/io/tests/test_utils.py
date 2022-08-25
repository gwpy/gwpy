# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""Unit tests for :mod:`gwpy.io.utils`
"""

import gzip
import tempfile
from pathlib import Path

import pytest

from .. import (
    cache as io_cache,
    utils as io_utils,
)

from .test_cache import cache  # noqa: F401

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def test_gopen(tmp_path):
    tmp = tmp_path / "test.tmp"
    # test simple use
    with tmp.open('w') as f:
        f.write('blah blah blah')
    with io_utils.gopen(tmp) as f2:
        assert f2.read() == 'blah blah blah'


@pytest.mark.parametrize("suffix", (".txt.gz", ""))
def test_gopen_gzip(tmp_path, suffix):
    tmp = tmp_path / f"test{suffix}"
    text = b'blah blah blah'
    with gzip.open(tmp, 'wb') as fobj:
        fobj.write(text)
    with io_utils.gopen(tmp, mode='rb') as fobj2:
        assert isinstance(fobj2, gzip.GzipFile)
        assert fobj2.read() == text


def test_identify_factory():
    id_func = io_utils.identify_factory('.blah', '.blah2')
    assert id_func(None, None, None) is False
    assert id_func(None, 'test.txt', None) is False
    assert id_func(None, 'test.blah', None) is True
    assert id_func(None, 'test.blah2', None) is True
    assert id_func(None, 'test.blah2x', None) is False


def test_file_list_file(cache):  # noqa: F811
    # test file -> [file.name]
    with tempfile.NamedTemporaryFile() as f:
        assert io_utils.file_list(f) == [f.name]


@pytest.mark.requires("lal")
def test_file_list_cache(cache):  # noqa: F811
    from lal.utils import CacheEntry
    # test CacheEntry -> [CacheEntry.path]
    lcache = list(map(CacheEntry.from_T050017, cache))
    assert io_utils.file_list(lcache[0]) == [cache[0]]

    # test cache object -> pfnlist
    assert io_utils.file_list(lcache) == cache

    # test cache file -> pfnlist()
    with tempfile.NamedTemporaryFile(suffix='.lcf', mode='w') as f:
        io_cache.write_cache(lcache, f)
        f.seek(0)
        assert io_utils.file_list(f.name) == cache


def test_file_list_str():
    # test comma-separated list -> list
    assert io_utils.file_list('A,B,C,D') == ['A', 'B', 'C', 'D']

    # test list -> list
    assert io_utils.file_list(['A', 'B', 'C', 'D']) == ['A', 'B', 'C', 'D']


def test_file_list_error():
    with pytest.raises(ValueError):
        io_utils.file_list(1)


@pytest.mark.parametrize(("input_", "expected"), (
    ("test.txt", "test.txt"),
    ("file:///test/path.txt", "/test/path.txt"),
    (Path("test.txt"), "test.txt"),
))
def test_file_path(input_, expected):
    assert io_utils.file_path(input_) == expected


def test_file_path_file():
    """Check that :func:`gwpy.io.utils.file_path` can handle open files
    """
    with tempfile.NamedTemporaryFile() as f:
        assert io_utils.file_path(f) == f.name


@pytest.mark.parametrize("badthing", (
    1,
    ["test.txt"],
))
def test_file_path_errors(badthing):
    """Check that :func:`gwpy.io.utils.file_path` fails when expected
    """
    with pytest.raises(ValueError) as exc:
        io_utils.file_path(badthing)
    assert str(exc.value).startswith("Cannot parse file name for")


@pytest.mark.requires("lal")
def test_file_path_cacheentry():
    from lal.utils import CacheEntry
    path = "/path/to/A-B-0-1.txt"
    assert io_utils.file_path(CacheEntry.from_T050017(path)) == path
