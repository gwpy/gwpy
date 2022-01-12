# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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

"""Tests for `gwpy.plot.segments`
"""

import pytest

import numpy

from matplotlib import rcParams
from matplotlib.colors import ColorConverter
from matplotlib.collections import PatchCollection

from ...segments import (Segment, SegmentList, SegmentListDict,
                         DataQualityFlag, DataQualityDict)
from ...time import to_gps
from .. import SegmentAxes
from ..segments import SegmentRectangle
from .test_axes import TestAxes as _TestAxes

# extract color cycle
COLOR_CONVERTER = ColorConverter()
COLOR_CYCLE = rcParams['axes.prop_cycle'].by_key()['color']
COLOR0 = COLOR_CONVERTER.to_rgba(COLOR_CYCLE[0])


class TestSegmentAxes(_TestAxes):
    AXES_CLASS = SegmentAxes

    @staticmethod
    @pytest.fixture()
    def segments():
        return SegmentList([Segment(0, 3), Segment(6, 7)])

    @staticmethod
    @pytest.fixture()
    def flag():
        known = SegmentList([Segment(0, 3), Segment(6, 7)])
        active = SegmentList([Segment(1, 2), Segment(3, 4), Segment(5, 7)])
        return DataQualityFlag(name='Test segments', known=known,
                               active=active)

    def test_plot_flag(self, ax, flag):
        c = ax.plot_flag(flag)
        assert c.get_label() == flag.texname
        assert len(ax.collections) == 2
        assert ax.collections[0] is c

        flag.isgood = False
        c = ax.plot_flag(flag)
        assert tuple(c.get_facecolors()[0]) == (1., 0., 0., 1.)

        c = ax.plot_flag(flag, known={'facecolor': 'black'})
        c = ax.plot_flag(flag, known='fancy')

    def test_plot_dict(self, ax, flag):
        dqd = DataQualityDict()
        dqd['a'] = flag
        dqd['b'] = flag

        colls = ax.plot_dict(dqd)
        assert len(colls) == len(dqd)
        assert all(isinstance(c, PatchCollection) for c in colls)
        assert colls[0].get_label() == 'a'
        assert colls[1].get_label() == 'b'

        colls = ax.plot_dict(dqd, label='name')
        assert colls[0].get_label() == 'Test segments'
        colls = ax.plot_dict(dqd, label='anything')
        assert colls[0].get_label() == 'anything'

    def test_plot_segmentlist(self, ax, segments):
        c = ax.plot_segmentlist(segments)
        assert isinstance(c, PatchCollection)
        assert numpy.isclose(ax.dataLim.x0, 0.)
        assert numpy.isclose(ax.dataLim.x1, 7.)
        assert len(c.get_paths()) == len(segments)
        assert ax.get_epoch() == segments[0][0]
        # test y
        p = ax.plot_segmentlist(segments).get_paths()[0].get_extents()
        assert p.y0 + p.height/2. == 1.
        p = ax.plot_segmentlist(segments, y=8).get_paths()[0].get_extents()
        assert p.y0 + p.height/2. == 8.
        # test kwargs
        c = ax.plot_segmentlist(segments, label='My segments',
                                rasterized=True)
        assert c.get_label() == 'My segments'
        assert c.get_rasterized() is True
        # test collection=False
        c = ax.plot_segmentlist(segments, collection=False, label='test')
        assert isinstance(c, list)
        assert not isinstance(c, PatchCollection)
        assert c[0].get_label() == 'test'
        assert c[1].get_label() == ''
        assert len(ax.patches) == len(segments)
        # test empty
        c = ax.plot_segmentlist(type(segments)())

    def test_plot_segmentlistdict(self, ax, segments):
        sld = SegmentListDict()
        sld['TEST'] = segments
        ax.plot(sld)

    def test_plot(self, ax, segments, flag):
        dqd = DataQualityDict(a=flag)
        ax.plot(segments)
        ax.plot(flag)
        ax.plot(dqd)
        ax.plot(flag, segments, dqd)

    def test_insetlabels(self, ax, segments):
        ax.plot(segments)
        ax.set_insetlabels(True)

    def test_fmt_data(self, ax):
        # just check that the LIGOTimeGPS repr is in place
        value = 1234567890.123
        assert ax.format_xdata(value) == str(to_gps(value))

    # -- disable tests from upstream

    def test_imshow(self):
        return NotImplemented


def test_segmentrectangle():
    patch = SegmentRectangle((1.1, 2.4), 10)
    assert patch.get_xy(), (1.1, 9.6)
    assert numpy.isclose(patch.get_height(), 0.8)
    assert numpy.isclose(patch.get_width(), 1.3)
    assert patch.get_facecolor() == COLOR0

    # check kwarg passing
    patch = SegmentRectangle((1.1, 2.4), 10, facecolor='red')
    assert patch.get_facecolor() == COLOR_CONVERTER.to_rgba('red')

    # check valign
    patch = SegmentRectangle((1.1, 2.4), 10, valign='top')
    assert patch.get_xy() == (1.1, 9.2)
    patch = SegmentRectangle((1.1, 2.4), 10, valign='bottom')
    assert patch.get_xy() == (1.1, 10.0)
    with pytest.raises(ValueError):
        patch = SegmentRectangle((0, 1), 0, valign='blah')
