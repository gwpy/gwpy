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

"""Tests for :mod:`gwpy.segments.segments`
"""

import pytest

import h5py

from astropy.table import Table

from ...testing.utils import (
    assert_segmentlist_equal,
    assert_table_equal
)
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

    def test_to_table(self, segmentlist):
        segtable = segmentlist.to_table()
        assert_table_equal(
            segtable,
            Table(
                rows=[
                    (0, 1, 2, 1),
                    (1, 3, 4, 1),
                    (2, 4, 6, 2),
                    (3, 8, 10, 2),
                ],
                names=('index', 'start', 'end', 'duration'),
            ),
        )

    # -- test I/O -------------------------------

    def test_read_write_segwizard(self, segmentlist, tmp_path):
        tmp = tmp_path / "segments.txt"
        # check write/read returns the same list
        segmentlist.write(tmp)
        sl2 = self.TEST_CLASS.read(tmp, coalesce=False)
        assert_segmentlist_equal(sl2, segmentlist)
        assert isinstance(sl2[0][0], LIGOTimeGPS)

        # check that coalesceing does what its supposed to
        c = type(segmentlist)(segmentlist).coalesce()
        sl2 = self.TEST_CLASS.read(str(tmp), coalesce=True)
        assert_segmentlist_equal(sl2, c)

        # check gpstype kwarg
        sl2 = self.TEST_CLASS.read(tmp, gpstype=float)
        assert isinstance(sl2[0][0], float)

    def test_read_write_segwizard_strict(self, tmp_path):
        tmp = tmp_path / "segments.txt"
        tmp.write_text("0 0 1 .5")
        with pytest.raises(ValueError):
            self.TEST_CLASS.read(tmp, strict=True, format='segwizard')
        sl = self.TEST_CLASS.read(tmp, strict=False, format='segwizard')
        assert_segmentlist_equal(sl, [(0, 1)])

    def test_read_write_segwizard_twocol(self, tmp_path):
        tmp = tmp_path / "segments.txt"
        tmp.write_text("0 1\n2 3")
        sl = self.TEST_CLASS.read(tmp, format='segwizard')
        assert_segmentlist_equal(sl, [(0, 1), (2, 3)])

    @pytest.mark.parametrize('ext', ('.hdf5', '.h5'))
    def test_read_write_hdf5(self, segmentlist, tmp_path, ext):
        tmp = tmp_path / "segments{}".format(ext)

        # check basic write/read with auto-path discovery
        segmentlist.write(tmp, 'test-segmentlist')
        sl2 = self.TEST_CLASS.read(tmp)
        assert_segmentlist_equal(sl2, segmentlist)
        assert isinstance(sl2[0][0], LIGOTimeGPS)

        sl2 = self.TEST_CLASS.read(tmp, path='test-segmentlist')
        assert_segmentlist_equal(sl2, segmentlist)

        # check we can read directly from the h5 object
        with h5py.File(tmp, "r") as h5f:
            sl2 = self.TEST_CLASS.read(h5f["test-segmentlist"])
            assert_segmentlist_equal(sl2, segmentlist)

        # check overwrite kwarg
        with pytest.raises(IOError):
            segmentlist.write(tmp, 'test-segmentlist')
        segmentlist.write(tmp, 'test-segmentlist', overwrite=True)

        # check gpstype kwarg
        sl2 = self.TEST_CLASS.read(tmp, gpstype=float)
        assert_segmentlist_equal(sl2, segmentlist)
        assert isinstance(sl2[0][0], float)
