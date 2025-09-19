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

"""Tests for :mod:`gwpy.segments.segments`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import pytest
from astropy.table import Table

from ...testing.errors import pytest_skip_flaky_network
from ...testing.utils import (
    TEST_DATA_PATH,
    assert_segmentlist_equal,
    assert_table_equal,
)
from ...time import LIGOTimeGPS
from .. import (
    Segment,
    SegmentList,
)

if TYPE_CHECKING:
    from typing import SupportsFloat

TEST_SEGWIZARD_FILE = TEST_DATA_PATH / "X1-GWPY_TEST_SEGMENTS-0-10.txt"
TEST_SEGWIZARD_URI = (
    "https://gitlab.com/gwpy/gwpy/-/raw/v3.0.10/"
    + TEST_SEGWIZARD_FILE.relative_to(
        TEST_DATA_PATH.parent.parent.parent,
    ).as_posix()
)


def _as_segmentlist(*segments: tuple[SupportsFloat, SupportsFloat]) -> SegmentList:
    """Return ``segments`` as a `SegmentList`."""
    return SegmentList([Segment(a, b) for a, b in segments])


# -- Segment -------------------------

class TestSegment:
    """Test `gwpy.segments.Segment`."""

    TEST_CLASS = Segment

    @pytest.fixture
    @classmethod
    def segment(cls):
        """Create a test segment (fixture)."""
        return cls.TEST_CLASS(1, 2)

    def test_start_end(self, segment):
        """Test the start and end properties."""
        assert segment.start == 1.
        assert segment.end == 2.

    def test_repr(self, segment):
        """Test ``repr(segment)``."""
        assert repr(segment) == "Segment(1, 2)"

    def test_str(self, segment):
        """Test ``str(segment)``."""
        assert str(segment) == "[1 ... 2)"


# -- SegmentList ---------------------

class TestSegmentList:
    """Test `gwpy.segments.SegmentList`."""

    TEST_CLASS = SegmentList
    ENTRY_CLASS = Segment

    @classmethod
    def create(cls, *segments):
        """Create a test segment list."""
        return cls.TEST_CLASS([cls.ENTRY_CLASS(a, b) for a, b in segments])

    @pytest.fixture
    @classmethod
    def segmentlist(cls):
        """Create a test segment list (fixture)."""
        return cls.create((1, 2), (3, 4), (4, 6), (8, 10))

    # -- test methods ----------------

    def test_extent(self, segmentlist):
        """Test `SegmentList.extent()` returns the right type."""
        extent = segmentlist.extent()
        assert isinstance(extent, self.ENTRY_CLASS)
        assert extent == Segment(1, 10)

    def test_coalesce(self):
        """Test `SegmentList.coalesce()`."""
        segmentlist = self.create((1, 2), (3, 4), (4, 5))
        c = segmentlist.coalesce()
        assert c is segmentlist
        assert_segmentlist_equal(c, _as_segmentlist((1, 2), (3, 5)))
        assert isinstance(c[0], self.ENTRY_CLASS)

    def test_to_table(self, segmentlist):
        """Test `SegmentList.to_table()`."""
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
                names=("index", "start", "end", "duration"),
            ),
        )

    # -- test I/O --------------------

    def test_read_write_segwizard(self, segmentlist, tmp_path):
        """Test writing and reading back a `SegmentList` in ASCII format."""
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
        """Test writing and reading back a `SegmentList` in strict ASCII format."""
        tmp = tmp_path / "segments.txt"
        tmp.write_text("0 0 1 .5")
        with pytest.raises(
            ValueError,
            match=r"incorrect duration \.5",
        ):
            self.TEST_CLASS.read(tmp, strict=True, format="segwizard")
        sl = self.TEST_CLASS.read(tmp, strict=False, format="segwizard")
        assert_segmentlist_equal(sl, _as_segmentlist((0, 1)))

    def test_read_write_segwizard_twocol(self, tmp_path):
        """Test writing and reading back a `SegmentList` in two-col ASCII format."""
        tmp = tmp_path / "segments.txt"
        tmp.write_text("0 1\n2 3")
        sl = self.TEST_CLASS.read(tmp, format="segwizard")
        assert_segmentlist_equal(sl, _as_segmentlist((0, 1), (2, 3)))

    @pytest.mark.parametrize("ext", [".hdf5", ".h5"])
    def test_read_write_hdf5(self, segmentlist, tmp_path, ext):
        """Test writing and reading back a `SegmentList` in HDF5 format."""
        tmp = tmp_path / f"segments{ext}"

        # check basic write/read with auto-path discovery
        segmentlist.write(tmp, "test-segmentlist")
        sl2 = self.TEST_CLASS.read(tmp)
        assert_segmentlist_equal(sl2, segmentlist)
        assert isinstance(sl2[0][0], LIGOTimeGPS)

        sl2 = self.TEST_CLASS.read(tmp, path="test-segmentlist")
        assert_segmentlist_equal(sl2, segmentlist)

        # check we can read directly from the h5 object
        with h5py.File(tmp, "r") as h5f:
            sl2 = self.TEST_CLASS.read(h5f["test-segmentlist"])
            assert_segmentlist_equal(sl2, segmentlist)

        # check overwrite kwarg
        with pytest.raises(
            OSError,
            match=r"File .* already exists",
        ):
            segmentlist.write(tmp, "test-segmentlist")
        segmentlist.write(tmp, "test-segmentlist", overwrite=True)

        # check gpstype kwarg
        sl2 = self.TEST_CLASS.read(tmp, gpstype=float)
        assert_segmentlist_equal(sl2, segmentlist)
        assert isinstance(sl2[0][0], float)

    @pytest_skip_flaky_network
    def test_read_remote(self):
        """Test that reading directly from a remote URI works."""
        local = self.TEST_CLASS.read(TEST_SEGWIZARD_FILE)
        remote = self.TEST_CLASS.read(TEST_SEGWIZARD_URI, cache=False)
        assert_segmentlist_equal(local, remote)
