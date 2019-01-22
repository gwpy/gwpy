# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""Tests for :mod:`gwpy.segments.segments`
"""

import os
import shutil
import tempfile

import pytest

from ...testing.utils import (assert_segmentlist_equal, TemporaryFilename)
from ...time import LIGOTimeGPS
from .. import (Segment, SegmentList)


# -- Segment ------------------------------------------------------------------

class TestSegment(object):
    TEST_CLASS = Segment

    @classmethod
    @pytest.fixture()
    def segment(cls):
        return cls.TEST_CLASS(1, 2)

    def test_start_end(self, segment):
        assert segment.start == 1.
        assert segment.end == 2.

    def test_repr(self, segment):
        assert repr(segment) == 'Segment(1, 2)'

    def test_str(self, segment):
        assert str(segment) == '[1 ... 2)'


# -- SegmentList --------------------------------------------------------------

class TestSegmentList(object):
    TEST_CLASS = SegmentList
    ENTRY_CLASS = Segment

    @classmethod
    def create(cls, *segments):
        return cls.TEST_CLASS([cls.ENTRY_CLASS(a, b) for a, b in segments])

    @classmethod
    @pytest.fixture()
    def segmentlist(cls):
        return cls.create((1, 2), (3, 4), (4, 6), (8, 10))

    # -- test methods ---------------------------

    def test_extent(self, segmentlist):
        """Test `gwpy.segments.SegmentList.extent returns the right type
        """
        extent = segmentlist.extent()
        assert isinstance(extent, self.ENTRY_CLASS)
        assert extent == Segment(1, 10)

    def test_coalesce(self):
        segmentlist = self.create((1, 2), (3, 4), (4, 5))
        c = segmentlist.coalesce()
        assert c is segmentlist
        assert_segmentlist_equal(c, [(1, 2), (3, 5)])
        assert isinstance(c[0], self.ENTRY_CLASS)

    # -- test I/O -------------------------------

    def test_read_write_segwizard(self, segmentlist):
        with TemporaryFilename(suffix='.txt') as tmp:
            # check write/read returns the same list
            segmentlist.write(tmp)
            sl2 = self.TEST_CLASS.read(tmp, coalesce=False)
            assert_segmentlist_equal(sl2, segmentlist)
            assert isinstance(sl2[0][0], LIGOTimeGPS)

            # check that coalesceing does what its supposed to
            c = type(segmentlist)(segmentlist).coalesce()
            sl2 = self.TEST_CLASS.read(tmp, coalesce=True)
            assert_segmentlist_equal(sl2, c)

            # check gpstype kwarg
            sl2 = self.TEST_CLASS.read(tmp, gpstype=float)
            assert isinstance(sl2[0][0], float)

    @pytest.mark.parametrize('ext', ('.hdf5', '.h5'))
    def test_read_write_hdf5(self, segmentlist, ext):
        with TemporaryFilename(suffix=ext) as fp:
            # check basic write/read with auto-path discovery
            segmentlist.write(fp, 'test-segmentlist')
            sl2 = self.TEST_CLASS.read(fp)
            assert_segmentlist_equal(sl2, segmentlist)
            assert isinstance(sl2[0][0], LIGOTimeGPS)

            sl2 = self.TEST_CLASS.read(fp, path='test-segmentlist')
            assert_segmentlist_equal(sl2, segmentlist)

            # check overwrite kwarg
            with pytest.raises(IOError):
                segmentlist.write(fp, 'test-segmentlist')
            segmentlist.write(fp, 'test-segmentlist', overwrite=True)

            # check gpstype kwarg
            sl2 = self.TEST_CLASS.read(fp, gpstype=float)
            assert_segmentlist_equal(sl2, segmentlist)
            assert isinstance(sl2[0][0], float)
