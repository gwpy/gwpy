# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2018-2022)
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

"""Unit tests for :mod:`gwpy.plot`
"""

import pytest

import numpy

from ...segments import (Segment, SegmentList)
from ...testing import utils
from ...types import (Series, Array2D)
from .. import Plot
from .utils import FigureTestBase

numpy.random.seed(0)


# -- test classes -------------------------------------------------------------

class TestPlot(FigureTestBase):
    FIGURE_CLASS = Plot

    def test_init(self):
        plot = self.FIGURE_CLASS(figsize=(4, 3), dpi=100)
        assert tuple(plot.get_size_inches()) == (4., 3.)
        assert plot.colorbars == []

    def test_init_empty(self):
        plot = self.FIGURE_CLASS(geometry=(2, 2))
        assert len(plot.axes) == 4
        assert plot.axes[-1].get_subplotspec().get_geometry() == (2, 2, 3, 3)

    def test_init_with_data(self):
        # list
        plot = self.FIGURE_CLASS([1, 2, 3, 4])
        assert len(plot.axes) == 1
        utils.assert_array_equal(plot.gca().lines[-1].get_ydata(),
                                 numpy.array([1, 2, 3, 4]))
        plot.close()

        # series
        a = Series(range(10), dx=.1)
        plot = self.FIGURE_CLASS(a)
        ax = plot.gca()
        line = ax.lines[0]
        assert len(plot.axes) == 1
        assert len(ax.lines) == 1
        utils.assert_array_equal(line.get_xdata(), a.xindex.value)
        utils.assert_array_equal(line.get_ydata(), a.value)
        plot.close()

        # two series
        b = Series(range(10), dx=.1)
        plot = self.FIGURE_CLASS(a, b)
        assert len(plot.axes) == 1
        assert len(plot.axes[0].lines) == 2
        plot.close()

        # two series on separate axes
        plot = self.FIGURE_CLASS(a, b, separate=True, sharex=True, sharey=True)
        assert len(plot.axes) == 2
        for i, ax in enumerate(plot.axes):
            assert ax.get_subplotspec().get_geometry() == (2, 1, i, i)
            assert len(ax.lines) == 1
        assert plot.axes[1]._sharex is plot.axes[0]
        plot.close()

        # Array2D with imshow
        array = Array2D(numpy.random.random((10, 10)), dx=.1, dy=.2)
        plot = self.FIGURE_CLASS(array, method='imshow')
        assert len(plot.axes[0].images) == 1
        image = plot.axes[0].images[0]
        utils.assert_array_equal(image.get_array(), array.value.T)
        plot.close()

    def test_save(self, fig, tmp_path):
        tmp = tmp_path / "plot.png"
        fig.save(str(tmp))
        assert tmp.is_file()

    def test_get_axes(self, fig):
        fig.add_subplot(2, 1, 1, projection='rectilinear')
        fig.add_subplot(2, 1, 2, projection='polar')
        assert fig.get_axes() == fig.axes
        assert fig.get_axes(projection='polar') == fig.axes[1:]

    def test_colorbar(self, fig):
        ax = fig.gca()
        array = Array2D(numpy.random.random((10, 10)), dx=.1, dy=.2)
        image = ax.imshow(array)
        cbar = fig.colorbar(vmin=2, vmax=4, fraction=0.)
        assert cbar.mappable is image
        assert cbar.mappable.get_clim() == (2., 4.)

    def test_add_segments_bar(self, fig):
        ax = fig.add_subplot(xscale='auto-gps', epoch=150)
        ax.set_xlim(100, 200)
        ax.set_xlabel('test')
        segs = SegmentList([Segment(10, 110), Segment(150, 400)])
        segax = fig.add_segments_bar(segs)
        assert segax._sharex is ax
        assert ax.get_xlabel() == ''
        for ax_ in (ax, segax):
            assert ax_.get_xlim() == (100, 200)
            assert ax_.get_epoch() == 150.

        # check that it works again
        segax = fig.add_segments_bar(segs, ax=ax)

        # check errors
        with pytest.raises(ValueError):
            fig.add_segments_bar(segs, location='left')
