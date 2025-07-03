# Copyright (c) 2018-2025 Cardiff University
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

"""Tests for `gwpy.plot.segments`."""

import numpy
import pytest
from matplotlib import rcParams
from matplotlib.collections import PatchCollection
from matplotlib.colors import ColorConverter

from ...segments import (
    DataQualityDict,
    DataQualityFlag,
    Segment,
    SegmentList,
    SegmentListDict,
)
from ...time import to_gps
from .. import SegmentAxes
from ..segments import SegmentRectangle
from .test_axes import TestAxes as _TestAxes

# extract color cycle
COLOR_CONVERTER = ColorConverter()
COLOR_CYCLE = rcParams["axes.prop_cycle"].by_key()["color"]
COLOR0 = COLOR_CONVERTER.to_rgba(COLOR_CYCLE[0])


class TestSegmentAxes(_TestAxes):
    """Tests for `SegmentAxes`."""

    AXES_CLASS = SegmentAxes

    @staticmethod
    @pytest.fixture
    def segments():
        """Create a `SegmentList` for use in tests."""
        return SegmentList([Segment(0, 3), Segment(6, 7)])

    @staticmethod
    @pytest.fixture
    def flag():
        """Create a `DataQualityFlag` for use in tests."""
        known = SegmentList([
            Segment(0, 3),
            Segment(6, 7),
        ])
        active = SegmentList([
            Segment(1, 2),
            Segment(3, 4),
            Segment(5, 7),
        ])
        return DataQualityFlag(
            name="Test segments",
            known=known,
            active=active,
        )

    def test_plot_flag(self, ax, flag):
        """Test `SegmentAxes.plot_flag`."""
        c = ax.plot_flag(flag)
        assert c.get_label() == flag.texname
        assert len(ax.collections) == 2
        assert ax.collections[0] is c

        flag.isgood = False
        c = ax.plot_flag(flag)
        assert tuple(c.get_facecolors()[0]) == (1., 0., 0., 1.)

        c = ax.plot_flag(flag, known={"facecolor": "black"})
        c = ax.plot_flag(flag, known="fancy")

    def test_plot_dict(self, ax, flag):
        """Test `SegmentAxes.plot_dict`."""
        dqd = DataQualityDict()
        dqd["a"] = flag
        dqd["b"] = flag

        colls = ax.plot_dict(dqd)
        assert len(colls) == len(dqd)
        assert all(isinstance(c, PatchCollection) for c in colls)
        assert colls[0].get_label() == "a"
        assert colls[1].get_label() == "b"

        colls = ax.plot_dict(dqd, label="name")
        assert colls[0].get_label() == "Test segments"
        colls = ax.plot_dict(dqd, label="anything")
        assert colls[0].get_label() == "anything"

    def test_plot_segmentlist(self, ax, segments):
        """Test `SegmentAxes.plot_segmentlist`."""
        coll = ax.plot_segmentlist(segments)
        assert isinstance(coll, PatchCollection)
        assert numpy.isclose(ax.dataLim.x0, 0.)
        assert numpy.isclose(ax.dataLim.x1, 7.)
        assert len(coll.get_paths()) == len(segments)
        assert ax.get_epoch() == segments[0][0]

    def test_plot_segmentlist_y(self, ax, segments):
        """Test `SegmentAxes.plot_segmentlist(..., y=N)`."""
        p = ax.plot_segmentlist(
            segments,
            valign="bottom",
        ).get_paths()[0].get_extents()
        assert p.y0 == 0.
        p = ax.plot_segmentlist(
            segments,
            valign="bottom",
            y=8,
        ).get_paths()[0].get_extents()
        assert p.y0 == 8.

    def test_plot_segmentlist_kwargs(self, ax, segments):
        """Test `SegmentAxes.plot_segmentlist` works with kwargs."""
        coll = ax.plot_segmentlist(
            segments,
            label="My segments",
            rasterized=True,
        )
        assert coll.get_label() == "My segments"
        assert coll.get_rasterized() is True

    def test_plot_segmentlist_collection_false(self, ax, segments):
        """Test `SegmentAxes.plot_segmentlist(..., collection=False)`."""
        coll = ax.plot_segmentlist(
            segments,
            collection=False,
            label="test",
        )
        assert isinstance(coll, list)
        assert not isinstance(coll, PatchCollection)
        assert coll[0].get_label() == "test"
        assert coll[1].get_label() == ""
        assert len(ax.patches) == len(segments)

    def test_plot_segmentlist_empty(self, ax):
        """Test that `SegmentAxes.plot_segmentlist` handles an empty segmentlist."""
        c = ax.plot_segmentlist(SegmentList())
        assert isinstance(c, PatchCollection)

    def test_plot_segmentlistdict(self, ax, segments):
        """Test `SegmentAxes.plot_segmentlistdict`."""
        sld = SegmentListDict()
        sld["TEST"] = segments
        ax.plot(sld)

    def test_plot(self, ax, segments, flag):  # type: ignore[override]
        """Test `SegmentAxes.plot`."""
        dqd = DataQualityDict(a=flag)
        ax.plot(segments)
        ax.plot(flag)
        ax.plot(dqd)
        ax.plot(flag, segments, dqd)

    def test_insetlabels(self, ax, segments):
        """Test `SegmentAxes.insetlabels` property."""
        ax.plot(segments)
        ax.set_insetlabels(True)

    def test_fmt_data(self, ax):
        """Test `SegmentAxes.format_xdata`."""
        # just check that the LIGOTimeGPS repr is in place
        value = 1234567890.123
        assert ax.format_xdata(value) == str(to_gps(value))

    # -- disable tests from upstream

    def test_imshow(self):  # type: ignore[override]
        """Test `SegmentAxes.imshow`."""
        pytest.skip(f"not implemented for {type(self).__name__}")


def test_segmentrectangle():
    """Test `SegmentRectangle`."""
    patch = SegmentRectangle((1.1, 2.4), 10)
    assert patch.get_xy(), (1.1, 9.6)
    assert numpy.isclose(patch.get_height(), 0.8)
    assert numpy.isclose(patch.get_width(), 1.3)
    assert patch.get_facecolor() == COLOR0


def test_segmentrectangle_kwargs():
    """Test `SegmentRectangle` handling of kwargs."""
    patch = SegmentRectangle(
        (1.1, 2.4),
        10,
        facecolor="red",
    )
    assert patch.get_facecolor() == COLOR_CONVERTER.to_rgba("red")


@pytest.mark.parametrize(("valign", "y"), [
    pytest.param("top", 9.2, id="top"),
    pytest.param("bottom", 10., id="bottom"),
])
def test_segmentrectangle_valign(valign, y):
    """Test `SegmentRectangle` handling of `valign`."""
    patch = SegmentRectangle((1.1, 2.4), 10, valign=valign)
    assert patch.get_xy() == (1.1, y)


def test_segmentrectangle_valign_error():
    """Test handling of invalid `valign` with `SegmentRectangle`."""
    with pytest.raises(
        ValueError,
        match="valign must be one of 'top', 'center', or 'bottom'",
    ):
        SegmentRectangle((0, 1), 0, valign="blah")  # type: ignore[arg-type]
