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

from matplotlib import rcParams
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from ...time import to_gps
from ...types import (Series, Array2D)
from ...testing import utils
from .. import Axes
from .utils import AxesTestBase

numpy.random.seed(0)


class TestAxes(AxesTestBase):
    AXES_CLASS = Axes

    def test_plot(self, ax):
        series = Series(range(10), dx=.1)
        lines = ax.plot(
            series,
            series * 2, 'k--',
            series.xindex, series, 'b-',
            [1, 2, 3], [4, 5, 6],
        )

        # check line 1 maps the series with default params
        line = lines[0]
        linex, liney = line.get_data()
        utils.assert_array_equal(linex, series.xindex.value)
        utils.assert_array_equal(liney, series.value)

        # check line 2 maps 2*series with specific params
        line = lines[1]
        linex, liney = line.get_data()
        utils.assert_array_equal(linex, series.xindex.value)
        utils.assert_array_equal(liney, series.value * 2)
        assert line.get_color() == 'k'
        assert line.get_linestyle() == '--'

        # check line 3
        line = lines[2]
        linex, liney = line.get_data()
        utils.assert_array_equal(linex, series.xindex.value)
        utils.assert_array_equal(liney, series.value)
        assert line.get_color() == 'b'
        assert line.get_linestyle() == '-'

        # check line 4
        line = lines[3]
        linex, liney = line.get_data()
        utils.assert_array_equal(linex, [1, 2, 3])
        utils.assert_array_equal(liney, [4, 5, 6])

    @pytest.mark.parametrize('sortbycolor', (False, True))
    def test_scatter(self, ax, sortbycolor):
        x = numpy.arange(10)
        y = numpy.arange(10)
        c = numpy.random.random(10)
        coll = ax.scatter(x, y, c=c, sortbycolor=sortbycolor)
        if sortbycolor:  # sort the colours now
            c = c[numpy.argsort(c)]
        # check that the colour array is in the right order
        utils.assert_array_equal(coll.get_array(), c)

    def test_scatter_c_sort(self, ax):
        x = numpy.arange(10)
        y = numpy.arange(10)
        c = numpy.random.random(10)
        with pytest.deprecated_call():
            coll = ax.scatter(x, y, c=c, c_sort=True)
        utils.assert_array_equal(coll.get_array(), c[numpy.argsort(c)])

    @pytest.mark.parametrize('sortbycolor', (False, True))
    def test_scatter_no_color(self, ax, sortbycolor):
        # check that c=None works
        x = numpy.arange(10)
        y = numpy.arange(10)
        ax.scatter(x, y, c=None, sortbycolor=sortbycolor)

    @pytest.mark.parametrize('sortbycolor', (False, True))
    def test_scatter_non_array(self, ax, sortbycolor):
        # check that using non-array data works
        ax.scatter([1], [1], c=[1], sortbycolor=sortbycolor)

    @pytest.mark.parametrize('sortbycolor', (False, True))
    def test_scatter_positional(self, ax, sortbycolor):
        # check that using positional arguments works
        x = numpy.arange(10)
        y = numpy.arange(10)
        s = numpy.random.random(10)
        c = numpy.random.random(10)
        ax.scatter(x, y, s, c, sortbycolor=sortbycolor)

    @pytest.mark.parametrize('sortbycolor', (False, True))
    def test_scatter_errors(self, ax, sortbycolor):
        # check that errors are handled properly
        x = 1
        y = 'B'
        c = 'something else'
        with pytest.raises(ValueError) as exc:
            ax.scatter(x, y, c=c, sortbycolor=sortbycolor)
        if sortbycolor:  # gwpy error
            msg = "Axes.scatter argument 'sortbycolor'"
        else:  # matplotlib error
            msg = "'c' argument must be a "
        assert str(exc.value).startswith(msg)

    def test_imshow(self, ax):
        # standard imshow call
        array = numpy.random.random((10, 10))
        image2 = ax.imshow(array)
        utils.assert_array_equal(image2.get_array(), array)
        assert image2.get_extent() == (-.5, array.shape[0]-.5,
                                       array.shape[1]-.5, -.5)

    def test_imshow_array2d(self, ax):
        # overloaded imshow call (Array2D)
        array = Array2D(numpy.random.random((10, 10)), dx=.1, dy=.2)
        image = ax.imshow(array)
        utils.assert_array_equal(image.get_array(), array.value.T)
        assert image.get_extent() == tuple(array.xspan) + tuple(array.yspan)

        # check log scale uses non-zero boundaries
        ax.clear()
        ax.set_xlim(.1, 1)
        ax.set_ylim(.1, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        image = ax.imshow(array)
        assert image.get_extent() == (1e-300, array.xspan[1],
                                      1e-300, array.yspan[1])

    def test_pcolormesh(self, ax):
        array = Array2D(numpy.random.random((10, 10)), dx=.1, dy=.2)
        ax.grid(True, which="both", axis="both")
        mesh = ax.pcolormesh(array)
        utils.assert_array_equal(mesh.get_array(), array.T.flatten())
        utils.assert_array_equal(mesh.get_paths()[-1].vertices[2],
                                 (array.xspan[1], array.yspan[1]))
        # check that restore_grid decorator did its job
        try:
            assert all((
                ax.xaxis._major_tick_kw["gridOn"],
                ax.xaxis._minor_tick_kw["gridOn"],
                ax.yaxis._major_tick_kw["gridOn"],
                ax.yaxis._minor_tick_kw["gridOn"],
            ))
        except KeyError:  # matplotlib < 3.3.3
            assert all((
                ax.xaxis._gridOnMajor,
                ax.xaxis._gridOnMinor,
                ax.yaxis._gridOnMajor,
                ax.yaxis._gridOnMinor,
            ))

    def test_hist(self, ax):
        x = numpy.random.random(100) + 1
        min_ = numpy.min(x)
        max_ = numpy.max(x)
        n, bins, patches = ax.hist(x, logbins=True, bins=10, weights=1.)
        utils.assert_allclose(
            bins, numpy.logspace(numpy.log10(min_), numpy.log10(max_),
                                 11, endpoint=True),
        )

    def test_hist_error(self, ax):
        """Test that `ax.hist` presents the right error message for empty data
        """
        with pytest.raises(ValueError) as exc:
            ax.hist([], logbins=True)
        assert str(exc.value).startswith('cannot generate log-spaced '
                                         'histogram bins')
        # assert it works if we give the range manually
        ax.hist([], logbins=True, range=(1, 100))

    def test_tile(self, ax):
        x = numpy.arange(10)
        y = numpy.arange(x.size)
        w = numpy.ones_like(x) * .8
        h = numpy.ones_like(x) * .8

        # check default tiling (without colour)
        coll = ax.tile(x, y, w, h, anchor='ll')
        assert isinstance(coll, PolyCollection)
        for i, path in enumerate(coll.get_paths()):
            utils.assert_array_equal(
                path.vertices,
                numpy.asarray([
                    (x[i], y[i]),
                    (x[i], y[i] + h[i]),
                    (x[i] + w[i], y[i] + h[i]),
                    (x[i] + w[i], y[i]),
                    (x[i], y[i]),
                ]),
            )

    @pytest.mark.parametrize("sortbycolor", (False, True))
    def test_tile_sortbycolor(self, ax, sortbycolor):
        x = numpy.arange(10)
        y = numpy.arange(x.size)
        w = numpy.ones_like(x) * .8
        h = numpy.ones_like(x) * .8

        # check colour works with sorting (by default)
        c = numpy.random.random(x.size)
        coll2 = ax.tile(x, y, w, h, color=c, sortbycolor=sortbycolor)
        if sortbycolor:
            utils.assert_array_equal(coll2.get_array(), numpy.sort(c))
        else:
            utils.assert_array_equal(coll2.get_array(), c)

    def test_tile_c_sort(self, ax):
        x = numpy.arange(10)
        y = numpy.arange(x.size)
        w = numpy.ones_like(x) * .8
        h = numpy.ones_like(x) * .8
        c = numpy.random.random(x.size)
        with pytest.deprecated_call():
            coll2 = ax.tile(x, y, w, h, color=c, c_sort=True)
        utils.assert_array_equal(coll2.get_array(), numpy.sort(c))

    @pytest.mark.parametrize("anchor", ("lr", "ul", "ur", "center"))
    def test_tile_anchors(self, ax, anchor):
        # check anchor parsing
        x = numpy.arange(10)
        y = numpy.arange(x.size)
        w = numpy.ones_like(x) * .8
        h = numpy.ones_like(x) * .8
        ax.tile(x, y, w, h, anchor=anchor)

    def test_tile_anchor_error(self, ax):
        x = numpy.arange(10)
        y = numpy.arange(x.size)
        w = numpy.ones_like(x) * .8
        h = numpy.ones_like(x) * .8
        with pytest.raises(ValueError):
            ax.tile(x, y, w, h, anchor='blah')

    @pytest.mark.parametrize('cb_kw', [
        {},  # our default
        {'use_axesgrid': True, 'fraction': 0.15},  # match matplotlib behaviour
        {'use_axesgrid': False, 'fraction': 0.15},  # matplotlib default
    ])
    def test_colorbar(self, ax, cb_kw):
        array = Array2D(numpy.random.random((10, 10)), dx=.1, dy=.2)
        mesh = ax.pcolormesh(array)
        cbar = ax.colorbar(vmin=2, vmax=4, **cb_kw)
        assert cbar.mappable is mesh
        assert cbar.mappable.get_clim() == (2., 4.)

    def test_legend(self, ax):
        ax.plot(numpy.arange(5), label='test')
        leg = ax.legend()
        lframe = leg.get_frame()
        assert lframe.get_linewidth() == rcParams['patch.linewidth']
        for line in leg.get_lines():
            assert line.get_linewidth() == 6.

    def test_legend_no_handler_map(self, ax):
        ax.plot(numpy.arange(5), label='test')
        leg = ax.legend(handler_map=None)
        for line in leg.get_lines():
            assert line.get_linewidth() == rcParams["lines.linewidth"]

    def test_plot_mmm(self, ax):
        mean_ = Series(numpy.random.random(10))
        min_ = mean_ * .5
        max_ = mean_ * 1.5

        a, b, c, d = ax.plot_mmm(mean_, min_, max_)
        for line in (a, b, c):
            assert isinstance(line, Line2D)
        assert isinstance(d, PolyCollection)
        assert len(ax.lines) == 3
        assert len(ax.collections) == 1

    def test_fmt_data(self, ax):
        value = 1234567890.123
        result = str(to_gps(value))
        assert ax.format_xdata(value) == (
            ax.xaxis.get_major_formatter().format_data_short(value)
        )
        ax.set_xscale('auto-gps')
        ax.set_yscale('auto-gps')
        assert ax.format_xdata(value) == result
        assert ax.format_ydata(value) == result

    def test_epoch(self, ax):
        ax.set_xscale('auto-gps')
        assert not ax.get_epoch()
        ax.set_epoch(12345)
        assert ax.get_epoch() == 12345.0
