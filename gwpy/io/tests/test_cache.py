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

"""Unit test for `io` module
"""

from __future__ import print_function

import tempfile
from copy import deepcopy

import numpy

import pytest

from ...segments import (Segment, SegmentList)
from ...tests.utils import (skip_missing_dependency, TemporaryFilename)
from .. import cache as io_cache

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

SEGMENTS = SegmentList(map(Segment, [
    (0, 1),
    (1, 2),
    (4, 5),
]))


# -- fixtures -----------------------------------------------------------------

@pytest.fixture
def cache():
    try:
        from lal.utils import CacheEntry
    except ImportError as e:
        pytest.skip(str(e))

    cache = []
    for seg in SEGMENTS:
        d = seg[1] - seg[0]
        f = 'A-B-%d-%d.tmp' % (seg[0], d)
        cache.append(CacheEntry.from_T050017(f, coltype=int))
    return cache


@pytest.fixture
def segments():
    return deepcopy(SEGMENTS)


@pytest.fixture
def tmpfile():
    with TemporaryFilename() as tmp:
        yield tmp


# -- tests --------------------------------------------------------------------

def test_read_write_cache(cache, tmpfile):
    with open(tmpfile, 'w') as f:
        io_cache.write_cache(cache, f)

    # read from fileobj
    with open(tmpfile) as f:
        c2 = io_cache.read_cache(tmpfile)
    assert cache == c2

    # write with file name
    io_cache.write_cache(cache, tmpfile)

    # read from file name
    c3 = io_cache.read_cache(tmpfile)
    assert cache == c3


@pytest.mark.parametrize('input_, result', [
    (None, False),
    ([], False),
    (['A-B-12345-6.txt'], True),
])
def test_is_cache(input_, result):
    assert io_cache.is_cache(input_) is result


@skip_missing_dependency('lal.utils')
def test_is_cache_lal():
    cache = [io_cache.CacheEntry.from_T050017('/tmp/A-B-12345-6.txt')]
    assert io_cache.is_cache(cache)
    assert not io_cache.is_cache(cache + [None])


@skip_missing_dependency('glue.lal')
def test_is_cache_glue():
    assert io_cache.is_cache(io_cache.Cache())

    # check ASCII file gets returned as False
    a = numpy.array([[1, 2], [3, 4]])
    with tempfile.TemporaryFile() as f:
        numpy.savetxt(f, a)
        f.seek(0)
        assert io_cache.is_cache(f) is False


@skip_missing_dependency('lal.utils')
def test_is_cache_file():

    # check file(path) is return as True if parsed as Cache
    e = io_cache.CacheEntry.from_T050017('/tmp/A-B-12345-6.txt')
    with tempfile.NamedTemporaryFile() as f:
        # empty file should return False
        assert io_cache.is_cache(f) is False
        assert io_cache.is_cache(f.name) is False

        # cache file should return True
        io_cache.write_cache([e], f)
        f.seek(0)
        assert io_cache.is_cache(f) is True
        assert io_cache.is_cache(f.name) is True


def test_is_cache_entry():
    assert io_cache.is_cache_entry('/tmp/A-B-12345-6.txt')
    assert not io_cache.is_cache_entry('random-file-name.blah')
    try:
        e = io_cache.CacheEntry.from_T050017('/tmp/A-B-12345-6.txt')
    except AttributeError:
        pass
    else:
        assert io_cache.is_cache_entry(e)


def test_file_list(cache):

    # test file -> [file.name]
    with tempfile.NamedTemporaryFile() as f:
        assert io_cache.file_list(f) == [f.name]

    # test CacheEntry -> [CacheEntry.path]
    assert io_cache.file_list(cache[0]) == [cache[0].path]

    # test cache file -> pfnlist()
    with tempfile.NamedTemporaryFile(suffix='.lcf', mode='w') as f:
        io_cache.write_cache(cache, f)
        f.seek(0)
        assert io_cache.file_list(f.name) == [e.path for e in cache]

    # test comma-separated list -> list
    assert io_cache.file_list('A,B,C,D') == ['A', 'B', 'C', 'D']

    # test cache object -> pfnlist
    assert io_cache.file_list(cache) == [e.path for e in cache]

    # test list -> list
    assert io_cache.file_list(['A', 'B', 'C', 'D']) == ['A', 'B', 'C', 'D']

    # otherwise error
    with pytest.raises(ValueError):
        io_cache.file_list(1)


def test_file_name(cache):

    # check file_name(<str>)
    assert io_cache.file_name('test.txt') == 'test.txt'

    # check file_name(<file>)
    with tempfile.NamedTemporaryFile() as f:
        assert io_cache.file_name(f) == f.name

    # check file_name(<CacheEntry>)
    assert io_cache.file_name(cache[0]) == cache[0].path

    # check that anything else fails
    with pytest.raises(ValueError):
        io_cache.file_name(1)
    with pytest.raises(ValueError):
        io_cache.file_name(['test.txt'])


def test_cache_segments(cache, segments):
    """Test :func:`gwpy.io.cache.cache_segments`
    """
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


def test_file_segment():
    """Test :func:`gwpy.io.cache.file_segment`
    """
    # check basic
    fs = io_cache.file_segment('A-B-1-2.ext')
    assert isinstance(fs, Segment)
    assert fs == Segment(1, 3)

    # check mutliple file extensions
    assert io_cache.file_segment('A-B-1-2.ext.gz') == (1, 3)

    # check floats (and multiple file extensions)
    assert io_cache.file_segment('A-B-1.23-4.ext.gz') == (1.23, 5.23)

    # test errors
    with pytest.raises(ValueError) as exc:
        io_cache.file_segment('blah')
    assert str(exc.value) == ('Failed to parse \'blah\' as '
                              'LIGO-T050017-compatible filename')


def test_flatten(cache):
    """Test :func:`gwpy.io.cache.flatten`
    """
    # check flattened version of single cache is unchanged
    assert io_cache.flatten(cache) == cache
    assert io_cache.flatten(cache, cache) == cache

    # check two caches get concatenated properly
    a = cache
    b = deepcopy(cache)
    for e in b:
        e.segment = e.segment.shift(10)
    c = a + b
    assert io_cache.flatten(a, b) == c


def test_find_contiguous(cache, segments):
    """Test :func:`gwpy.io.cache.find_contiguous`
    """
    for i, cache in enumerate(io_cache.find_contiguous(cache)):
        io_cache.cache_segments(cache).extent() == segments[i]

    assert not list(io_cache.find_contiguous())


def test_sieve(cache, segments):
    sieved = io_cache.sieve(cache, segments[0])
    assert type(sieved) is type(cache)
    assert sieved == cache[:1]

    segments.coalesce()
    assert io_cache.sieve(cache, segments[0]) == cache[:2]
