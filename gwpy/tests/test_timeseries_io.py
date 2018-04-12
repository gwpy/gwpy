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

"""Unit tests for `timeseries.io` module
"""

from gwpy.segments import (Segment, SegmentList)
from gwpy.timeseries.io import cache as tio_cache

from utils import assert_segmentlist_equal

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def test_get_mp_cache_segments():
    """Test `gwpy.timeseries.io.cache.get_mp_cache_segments`
    """
    from lal.utils import CacheEntry
    from glue.lal import Cache
    from glue.segmentsUtils import segmentlist_range
    Cache.entry_class = CacheEntry

    # make cache
    cache = Cache()
    segments = SegmentList([Segment(0, 10), Segment(20, 30)])
    fsegs = SegmentList([s for seg in segments for
                         s in segmentlist_range(seg[0], seg[1], 2)])
    cache = Cache([CacheEntry.from_T050017(
                       'A-B-{0}-{1}.ext'.format(s[0], abs(s)))
                   for s in fsegs])

    # assert that no multiprocessing just returns the segment
    assert_segmentlist_equal(
        tio_cache.get_mp_cache_segments(cache, 1, Segment(0, 30)),
        SegmentList([Segment(0, 30)]))

    # simple test that segments get divided as expected
    mpsegs = tio_cache.get_mp_cache_segments(cache, 2, Segment(0, 30))
    assert_segmentlist_equal(mpsegs, segments)

    # test that mismatch with files edges is fine
    mpsegs = tio_cache.get_mp_cache_segments(cache, 2, Segment(0, 21))
    assert not mpsegs - SegmentList([Segment(0, 21)])

    # test segment divisions
    mpsegs = tio_cache.get_mp_cache_segments(cache, 4, Segment(0, 30))
    assert_segmentlist_equal(
        mpsegs,
        SegmentList([s for seg in segments for
                     s in segmentlist_range(s[0], s[1], 5)]),
    )
