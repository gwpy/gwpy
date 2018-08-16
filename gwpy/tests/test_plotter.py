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

"""Unit tests for plotter module
"""

import warnings
from io import BytesIO

import pytest

import numpy
from numpy import testing as nptest

from scipy import signal

from matplotlib import (use, rc_context, __version__ as mpl_version)
use('agg')  # nopep8
from matplotlib import (pyplot, rcParams)
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.colors import (LogNorm, ColorConverter)
from matplotlib.collections import (PathCollection, PatchCollection,
                                    PolyCollection)

from astropy import units

from gwpy.segments import (DataQualityFlag,
                           Segment, SegmentList, SegmentListDict)
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from gwpy.table import EventTable
from gwpy.plotter import (figure, Plot, Axes,
                          TimeSeriesPlot, TimeSeriesAxes,
                          FrequencySeriesPlot, FrequencySeriesAxes,
                          EventTablePlot, EventTableAxes,
                          HistogramPlot, HistogramAxes,
                          SegmentPlot, SegmentAxes,
                          SpectrogramPlot, BodePlot)
from gwpy.plotter.rc import (SUBPLOT_WIDTH, SUBPLOT_HEIGHT)
from gwpy.plotter.gps import (GPSTransform, InvertedGPSTransform)
from gwpy.plotter.html import (map_data, map_artist)
from gwpy.plotter.log import CombinedLogFormatterMathtext
from gwpy.plotter.text import (to_string, unit_as_label)
from gwpy.plotter.tex import (float_to_latex, label_to_latex,
                              unit_to_latex, HAS_TEX)
from gwpy.plotter.table import get_column_string

from . import utils

# ignore matplotlib complaining about GUIs
warnings.filterwarnings(
    'ignore', category=UserWarning, message=".*non-GUI backend.*")

# design ZPK for BodePlot test
ZPK = [100], [1], 1e-2
FREQUENCIES, MAGNITUDE, PHASE = signal.bode(ZPK, n=100)

# extract color cycle
COLOR_CONVERTER = ColorConverter()
try:
    COLOR_CYCLE = rcParams['axes.prop_cycle'].by_key()['color']
except KeyError:  # mpl < 1.5
    COLOR0 = COLOR_CONVERTER.to_rgba('b')
else:
    if mpl_version >= '2.0':
        COLOR0 = COLOR_CONVERTER.to_rgba(COLOR_CYCLE[0])
    else:
        COLOR0 = COLOR_CONVERTER.to_rgba('b')

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@pytest.fixture(scope='function', params=[
    pytest.param(False, id='usetex'),
    pytest.param(True, id='no-tex', marks=pytest.mark.skipif(
        not HAS_TEX, reason='no latex')),
])
def usetex(request):
    """Fixture to test plotting function with and without `usetex`

    Returns
    -------
    usetex : `bool`
        the value of the `text.usetex` rcParams settings
    """
    use_ = request.param
    with rc_context(rc={'text.usetex': use_}):
        yield use_


class PlottingTestBase(object):
    FIGURE_CLASS = Plot
    AXES_CLASS = Axes

    def new(self, **figkwargs):
        """Create a new `Figure` with some `Axes`

        Returns (fig, ax)
        """
        fig = self.FIGURE_CLASS(**figkwargs)
        return fig, fig.gca()

    @property
    def use_tex(self):
        return rcParams['text.usetex']

    def save(self, fig, suffix='.png'):
        fig.save(BytesIO(), format=suffix.lstrip('.'))
        return fig

    def save_and_close(self, fig, suffix='.png'):
        self.save(fig, suffix=suffix)
        fig.close()
        return fig


class TestPlot(PlottingTestBase):
    def test_init(self):
        # test object creation
        fig, ax = self.new()
        assert isinstance(fig, self.FIGURE_CLASS)

        # test added properties
        assert isinstance(fig.colorbars, list)
        assert len(fig.colorbars) == 0
        assert isinstance(fig._coloraxes, list)
        assert len(fig._coloraxes) == 0

        # test gca
        assert isinstance(ax, self.AXES_CLASS)
        self.save_and_close(fig)

    # -- plot rendering -------------------------

    def test_figure(self):
        fig = figure()
        assert isinstance(fig, Plot)
        fig.close()
        fig = figure(FigureClass=self.FIGURE_CLASS)
        assert isinstance(fig, self.FIGURE_CLASS)
        fig.close()

    def test_close(self):
        fig = self.FIGURE_CLASS()
        assert fig.canvas.manager.num in pyplot.get_fignums()
        fig.close()
        assert fig.canvas.manager.num not in pyplot.get_fignums()

    def test_show(self):
        # no idea how to assert that this worked
        fig = self.FIGURE_CLASS()
        fig.show(block=True, warn=False)
        fig.show(block=False, warn=False)

    def test_refresh(self):
        # no idea how to assert that this worked
        fig, ax = self.new()
        fig.refresh()

    def test_auto_refresh(self):
        fig = self.FIGURE_CLASS()
        assert fig.get_auto_refresh() is False

        fig.set_auto_refresh(True)
        assert fig.get_auto_refresh() is True

        fig = self.FIGURE_CLASS(auto_refresh=True)
        assert fig.get_auto_refresh()
        self.save_and_close(fig)

    def test_subplotpars(self):
        # check that dynamic subplotpars gets applied
        fig, ax = self.new(figsize=(12, 4))
        target = SUBPLOT_WIDTH[12] + SUBPLOT_HEIGHT[4]
        sbp = fig.subplotpars
        assert target == (sbp.left, sbp.right, sbp.bottom, sbp.top)

        # check that dynamic subplotpars doesn't get applied if the user
        # overrides any of the settings
        with rc_context(rc={'figure.subplot.left': target[0]*.1}):
            fig, ax = self.new(figsize=(12, 4))
            sbp = fig.subplotpars
            assert sbp.left, target[0]*.1

    # -- plot manipulation ----------------------

    @pytest.mark.parametrize('name, args', [
        ('xlim', (4, 5)),
        ('xlabel', 'test'),
        ('xscale', 'log'),
        ('ylim', (4, 5)),
        ('ylabel', 'test'),
        ('yscale', 'log'),
        ('title', 'test'),
    ])
    def test_axes_methods(self, name, args):
        fig, ax = self.new()

        fig_get = getattr(fig, 'get_%s' % name)
        fig_set = getattr(fig, 'set_%s' % name)
        ax_get = getattr(ax, 'get_%s' % name)

        # note: @axes_methods is deprecated
        with pytest.warns(DeprecationWarning):
            assert fig_get() == ax_get()
        with pytest.warns(DeprecationWarning):
            fig_set(args)
        with pytest.warns(DeprecationWarning):
            assert fig_get() == args
        with pytest.warns(DeprecationWarning):
            assert fig_get() == ax_get()

    @pytest.mark.parametrize('axis', ('x', 'y'))
    def test_log(self, axis):
        fig, ax = self.new()

        # fig.set_xlim(0.1, 10)
        with pytest.warns(DeprecationWarning):
            getattr(fig, 'set_%slim' % axis)(0.1, 10)

        # fig.logx = True
        with pytest.warns(DeprecationWarning):
            setattr(fig, 'log%s' % axis, True)

        # assert ax.get_xscale() == 'log'
        assert getattr(ax, 'get_%sscale' % axis)() == 'log'

        # assert fig.logx is True
        with pytest.warns(DeprecationWarning):
            assert getattr(fig, 'log%s' % axis) is True

        # fig.logx = False
        with pytest.warns(DeprecationWarning):
            setattr(fig, 'log%s' % axis, False)
        assert getattr(ax, 'get_%sscale' % axis)() == 'linear'
        with pytest.warns(DeprecationWarning):
            assert getattr(fig, 'log%s' % axis) is False

        self.save_and_close(fig)

    # -- artist creation ------------------------

    def test_add_legend(self):
        fig, ax = self.new()
        ax.plot([1, 2, 3, 4], label='Plot')
        assert isinstance(fig.add_legend(), Legend)
        self.save_and_close(fig)

    def test_add_line(self):
        fig, ax = self.new()
        fig.add_line([1, 2, 3, 4], [1, 2, 3, 4])
        assert len(ax.lines) == 1
        fig.close()

    def test_add_scatter(self):
        fig, ax = self.new()
        fig.add_scatter([1, 2, 3, 4], [1, 2, 3, 4])
        assert len(ax.collections) == 1
        fig.close()

    def test_add_image(self):
        fig, ax = self.new()
        data = numpy.arange(120).reshape((10, 12))
        image = fig.add_image(data, cmap='Blues')
        assert ax.images == [image]
        assert image.get_cmap().name == 'Blues'

        fig = self.FIGURE_CLASS()
        fig.add_image(data)

    def test_add_arrays(self):
        ts = TimeSeries([1, 2, 3, 4])
        fs = FrequencySeries([1, 2, 3, 4])
        fig = self.FIGURE_CLASS()
        assert len(fig.axes) == 0

        fig.add_timeseries(ts)
        assert len(fig.axes) == 1
        assert isinstance(fig.axes[0], TimeSeriesAxes)

        fig.add_frequencyseries(fs)
        assert len(fig.axes) == 2
        assert isinstance(fig.axes[1], FrequencySeriesAxes)
        fig.close()

    def test_add_colorbar(self):
        fig = self.FIGURE_CLASS()

        # test that adding a colorbar to an empty plot fails
        with pytest.raises(ValueError) as exc:
            fig.add_colorbar()
        assert str(exc.value) == ('Cannot determine mappable layer for '
                                  'this colorbar')
        with pytest.raises(ValueError) as exc:
            fig.add_colorbar(visible=False)
        assert str(exc.value) == ('Cannot determine an anchor Axes for '
                                  'this colorbar')

        # add axes and an image
        ax = fig.gca()
        width = ax.get_position().width
        mappable = ax.imshow(numpy.arange(120).reshape((10, 12)))
        assert not isinstance(mappable.norm, LogNorm)

        # add colorbar and check everything went through
        cbar = fig.add_colorbar(log=True, label='Test label', cmap='YlOrRd')
        assert len(fig.axes) == 2
        assert cbar in fig.colorbars
        assert cbar.ax in fig._coloraxes
        assert cbar.mappable is mappable
        assert cbar.get_clim() == mappable.get_clim() == (1, 119)
        assert isinstance(mappable.norm, LogNorm)
        assert isinstance(cbar.formatter, CombinedLogFormatterMathtext)
        assert cbar.get_cmap().name == 'YlOrRd'
        assert cbar.ax.get_ylabel() == 'Test label'
        self.save_and_close(fig)
        assert ax.get_position().width < width

        # try a non-visible colorbar
        fig = self.FIGURE_CLASS()
        ax = fig.gca()
        assert len(fig.axes) == 1
        fig.add_colorbar(ax=ax, visible=False)
        assert len(fig.axes) == 1
        fig.close()

        # check errors
        mappable = ax.imshow(numpy.arange(120).reshape((10, 12)))
        with pytest.raises(ValueError):
            fig.add_colorbar(mappable=mappable, location='bottom')


class TestAxes(PlottingTestBase):

    # -- test properties ------------------------
    # all of these properties are DEPRECATED

    @pytest.mark.parametrize('axis', ('x', 'y'))
    def test_label(self, axis):
        fig, ax = self.new()

        axis_obj = getattr(ax, '%saxis' % axis)
        label = '%slabel' % axis
        get_label = getattr(ax, 'get_%slabel' % axis)

        # assert ax.xlabel is ax.xaxis.label
        with pytest.warns(DeprecationWarning):
            assert getattr(ax, label) is axis_obj.label

        # ax.xlabel = 'Test label'
        with pytest.warns(DeprecationWarning):
            setattr(ax, label, 'Test label')
        # assert ax.get_xlabel() == 'Test label'
        assert get_label() == 'Test label'

        # check Text object gets preserved
        t = ax.text(0, 0, 'Test text')
        with pytest.warns(DeprecationWarning):
            setattr(ax, label, t)
        assert axis_obj.label is t

        # check deleter works
        with pytest.warns(DeprecationWarning):
            delattr(ax, label)
        assert get_label() == ''

    @pytest.mark.parametrize('axis', ('x', 'y'))
    def test_lim(self, axis):
        fig, ax = self.new()
        ax.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        lim = '%slim' % axis
        get_lim = getattr(ax, 'get_%slim' % axis)

        # check getter/setter
        with pytest.warns(DeprecationWarning):
            setattr(ax, lim, (24, 36))
        assert get_lim() == (24, 36)
        with pytest.warns(DeprecationWarning):
            assert getattr(ax, lim) == get_lim()

        # check deleter
        with pytest.warns(DeprecationWarning):
            delattr(ax, lim)

    @pytest.mark.parametrize('axis', ('x', 'y'))
    def test_log(self, axis):
        fig, ax = self.new()
        log = 'log%s' % axis
        get_scale = getattr(ax, 'get_%sscale' % axis)
        set_scale = getattr(ax, 'set_%sscale' % axis)

        # check default is not log
        with pytest.warns(DeprecationWarning):
            assert getattr(ax, log) is False

        # set log and assert that the scale gets set properly
        with pytest.warns(DeprecationWarning):
            setattr(ax, log, True)
        with pytest.warns(DeprecationWarning):
            assert getattr(ax, log) is True
        assert get_scale() == 'log'

        # set not log and check
        with pytest.warns(DeprecationWarning):
            setattr(ax, log, False)
        with pytest.warns(DeprecationWarning):
            assert getattr(ax, log) is False
        assert get_scale() == 'linear'

    # -- test methods ---------------------------

    def test_resize(self):
        fig, ax = self.new()
        ax.resize((0.25, 0.25, 0.5, 0.5))
        assert ax.get_position().bounds == (.25, .25, .5, .5)

    def test_legend(self):
        fig = self.FIGURE_CLASS()
        ax = fig.add_subplot(111, projection=self.AXES_CLASS.name)
        ax.plot([1, 2, 3, 4], label='Plot')

        leg = ax.legend()
        legframe = leg.get_frame()
        assert legframe.get_alpha() == 0.8
        assert legframe.get_linewidth() == rcParams['axes.linewidth']

        for l in leg.get_lines():
            assert l.get_linewidth() == 8
        self.save_and_close(fig)

    def test_html_map(self):
        # this method just runs the html_map method but puts little effort
        # into validating the result, that is left for the TestHtml
        # suite

        fig, ax = self.new()

        with pytest.raises(ValueError) as exc:
            ax.html_map('test.png')
        assert str(exc.value) == 'Cannot determine artist to map, 0 found.'

        line = ax.plot([1, 2, 3, 4, 5])[0]

        # auto-detect artist
        hmap = ax.html_map('test.png')
        assert hmap.startswith('<!doctype html>')
        assert hmap.count('<area') == 5

        # manually pass artist
        hmap2 = ax.html_map('test.png', data=line)
        assert hmap2 == hmap

        # auto-detect with multiple artists
        ax.plot([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            ax.html_map('test.png')

        # check data=<array>
        data = list(zip(*line.get_data()))
        hmap2 = ax.html_map('test.png', data=data)
        assert hmap2 == hmap

    @pytest.mark.parametrize('method', [
        'plot',
        'loglog',
        'semilogx',
        'semilogy',
    ])
    def test_matplotlib_compat(self, method):
        fig, ax = self.new()
        plot = getattr(ax, method)
        plot([1, 2, 3, 4, 5])
        plot([1, 2, 3, 4, 5], color='green', linestyle='--')
        self.save_and_close(fig)


# -- TimeSeries plotters ------------------------------------------------------

class TimeSeriesMixin(object):
    FIGURE_CLASS = TimeSeriesPlot
    AXES_CLASS = TimeSeriesAxes

    @classmethod
    def setup_class(cls):
        numpy.random.seed(0)
        cls.ts = TimeSeries(numpy.random.rand(10000), sample_rate=128)
        cls.sg = cls.ts.spectrogram2(0.5, 0.49)
        cls.mmm = [cls.ts, cls.ts * 0.9, cls.ts*1.1]


class TestTimeSeriesPlot(TimeSeriesMixin, TestPlot):
    def test_init(self):
        # test empty
        fig, ax = self.new()
        assert isinstance(ax, self.AXES_CLASS)

        # test passing arguments
        fig = self.FIGURE_CLASS(figsize=[9, 6])
        assert fig.get_figwidth() == 9
        fig = self.FIGURE_CLASS(self.ts)
        ax = fig.gca()
        assert len(ax.lines) == 1
        assert ax.get_xlim() == self.ts.span

        # test passing multiple timeseries
        fig = self.FIGURE_CLASS(self.ts, self.ts)
        assert len(fig.gca().lines) == 2
        fig = self.FIGURE_CLASS(self.ts, self.ts, sep=True, sharex=True)
        assert len(fig.axes) == 2
        for ax in fig.axes:
            assert len(ax.lines) == 1
        assert fig.axes[1]._sharex is fig.axes[0]

        # test kwarg parsing
        fig = self.FIGURE_CLASS(self.ts, figsize=[12, 6], rasterized=True)

    def test_add_colorbar(self):
        def make_fig():
            fig = self.FIGURE_CLASS(self.sg ** (1/2.))
            return fig, fig.gca()

        # test basic
        fig, ax = make_fig()
        cb = fig.add_colorbar()
        assert len(fig.colorbars) == 1
        assert cb is fig.colorbars[0]
        self.save_and_close(fig)
        # test kwargs
        fig, ax = make_fig()
        fig.add_colorbar(ax=ax, norm='log', clim=[1e-22, 1e-18],
                         label='Test colorbar')
        self.save_and_close(fig)

    def test_add_state_segments(self):
        fig, ax = self.new()

        # mock up some segments and add them as 'state' segments
        segs = SegmentList([Segment(1, 2), Segment(4, 5)])
        segax = fig.add_state_segments(segs)

        # check that the new axes aligns with the parent
        utils.assert_array_equal(segax.get_position().intervalx,
                                 ax.get_position().intervalx)
        coll = segax.collections[0]
        for seg, path in zip(segs, coll.get_paths()):
            utils.assert_array_equal(
                path.vertices, [(seg[0], -.4), (seg[1], -.4), (seg[1], .4),
                                (seg[0], .4), (seg[0], -.4)])

        with pytest.raises(ValueError):
            fig.add_state_segments(segs, location='left')

        # test that this doesn't work with non-timeseries axes
        fig = self.FIGURE_CLASS()
        ax = fig.gca(projection='rectilinear')
        with pytest.raises(ValueError) as exc:
            fig.add_state_segments(segs)
        assert str(exc.value) == ("No 'timeseries' Axes found, cannot anchor "
                                  "new segment Axes.")


class TestTimeSeriesAxes(TimeSeriesMixin, TestAxes):
    def test_init(self):
        fig, ax = self.new()
        assert isinstance(ax, self.AXES_CLASS)
        assert ax.get_xscale() == 'auto-gps'
        assert ax.get_xlabel() == '_auto'
        self.save_and_close(fig)

    def test_plot_timeseries(self):
        fig, ax = self.new()
        # check method works
        ax.plot_timeseries(self.ts)
        # check data are correct
        line = ax.get_lines()[0]
        nptest.assert_array_equal(line.get_xdata(), self.ts.times.value)
        nptest.assert_array_equal(line.get_ydata(), self.ts.value)
        # check GPS axis is set ok
        assert ax.get_xlim() == tuple(self.ts.span)
        self.save_and_close(fig)

    def test_plot_mmm(self):
        fig, ax = self.new()
        # test default
        a, b, c, d, e = ax.plot_mmm(*self.mmm)
        for line in (a, b, d):
            assert isinstance(line, Line2D)
        for coll in (c, e):
            assert isinstance(coll, PolyCollection)
        assert len(ax.lines) == 3
        assert len(ax.collections) == 2
        self.save_and_close(fig)

        # test with labels
        fig, ax = self.new()
        minname = self.mmm[1].name
        maxname = self.mmm[2].name
        self.mmm[1].name = 'min'
        self.mmm[2].name = 'max'
        try:
            ax.plot_mmm(*self.mmm, label='test')
            leg = ax.legend()
            assert len(leg.get_lines()) == 1
        finally:
            self.mmm[1].name = minname
            self.mmm[2].name = maxname
        self.save_and_close(fig)

        # test min only
        fig, ax = self.new()
        a, b, c, d, e = ax.plot_mmm(self.mmm[0], min_=self.mmm[1])
        assert d is None
        assert e is None
        assert len(ax.lines) == 2
        assert len(ax.collections) == 1
        self.save_and_close(fig)

        # test max only
        fig, ax = self.new()
        a, b, c, d, e = ax.plot_mmm(self.mmm[0], max_=self.mmm[2])
        assert b is None
        assert c is None
        assert len(ax.lines) == 2
        assert len(ax.collections) == 1

    def test_plot_timeseries_mmm(self):
        fig, ax = self.new()
        with pytest.warns(DeprecationWarning):
            ax.plot_timeseries_mmm(*self.mmm)
        self.save_and_close(fig)

    def test_plot_spectrogram(self):
        fig, ax = self.new()
        # check method
        ax.plot_spectrogram(self.sg, imshow=False)
        coll = ax.collections[0]
        nptest.assert_array_equal(coll.get_array(), self.sg.value.T.flatten())
        # check GPS axis is set ok
        assert ax.get_xlim() == tuple(self.sg.xspan)
        # check frequency axis
        if self.use_tex:
            assert ax.get_ylabel() == r'Frequency [$\mathrm{Hz}$]'
        else:
            assert ax.get_ylabel() == r'Frequency [Hz]'
        # check kwarg parsing
        c = ax.plot_spectrogram(self.sg, norm='log')
        assert isinstance(c.norm, LogNorm)
        self.save_and_close(fig)

    def test_plot(self):
        fig, ax = self.new()
        ax.plot(self.ts)
        self.save_and_close(fig)


# -- FrequencySeries plotters -------------------------------------------------

class FrequencySeriesMixin(object):
    FIGURE_CLASS = FrequencySeriesPlot
    AXES_CLASS = FrequencySeriesAxes

    @classmethod
    def setup_class(cls):
        numpy.random.seed(0)
        cls.ts = TimeSeries(numpy.random.rand(10000), sample_rate=128)
        cls.asd = cls.ts.asd(1)
        cls.mmm = [cls.asd, cls.asd*0.9, cls.asd*1.1]


class TestFrequencySeriesPlot(FrequencySeriesMixin, TestPlot):
    def test_init(self):
        super(TestFrequencySeriesPlot, self).test_init()
        # test convenience plotting
        fig = self.FIGURE_CLASS(self.asd)
        assert len(fig.axes) == 1
        ax = fig.gca()
        assert isinstance(ax, self.AXES_CLASS)
        assert len(ax.lines) == 1


class TestFrequencySeriesAxes(FrequencySeriesMixin, TestAxes):
    def test_plot_frequencyseries(self):
        fig, ax = self.new()
        line = ax.plot_frequencyseries(self.asd)[0]
        nptest.assert_array_equal(line.get_xdata(), self.asd.frequencies.value)
        nptest.assert_array_equal(line.get_ydata(), self.asd.value)
        self.save_and_close(fig)

    def test_plot_mmm(self):
        fig, ax = self.new()
        # test defaults
        a, b, c, d, e = ax.plot_mmm(*self.mmm)
        for line in (a, b, d):
            assert isinstance(line, Line2D)
        for coll in (c, e):
            assert isinstance(coll, PolyCollection)
        assert len(ax.lines) == 3
        assert len(ax.collections) == 2
        self.save_and_close(fig)

        # test with labels
        fig, ax = self.new()
        minname = self.mmm[1].name
        maxname = self.mmm[2].name
        self.mmm[1].name = 'min'
        self.mmm[2].name = 'max'
        try:
            ax.plot_mmm(*self.mmm, label='test')
            leg = ax.legend()
            assert len(leg.get_lines()) == 1
        finally:
            self.mmm[1].name = minname
            self.mmm[2].name = maxname
        self.save_and_close(fig)

        # test min only
        fig, ax = self.new()
        a, b, c, d, e = ax.plot_mmm(self.mmm[0], min_=self.mmm[1])
        assert d is None
        assert e is None
        assert len(ax.lines) == 2
        assert len(ax.collections) == 1
        self.save_and_close(fig)

        # test max only
        fig, ax = self.new()
        a, b, c, d, e = ax.plot_mmm(self.mmm[0], max_=self.mmm[2])
        assert b is None
        assert c is None
        assert len(ax.lines) == 2
        assert len(ax.collections) == 1
        self.save_and_close(fig)

    def test_plot_frequencyseries_mmm(self):
        fig, ax = self.new()
        with pytest.warns(DeprecationWarning):
            ax.plot_frequencyseries_mmm(*self.mmm)
        self.save_and_close(fig)


# -- Table plotters -----------------------------------------------------------

class EventTableMixin(object):
    FIGURE_CLASS = EventTablePlot
    AXES_CLASS = EventTableAxes

    @classmethod
    def create(cls, n, names, dtypes=None):
        data = []
        for i in range(len(names)):
            numpy.random.seed(i)
            if dtypes:
                dtype = dtypes[i]
            else:
                dtype = None
            data.append((numpy.random.rand(n) * 1000).astype(dtype))
        return EventTable(data, names=names)

    @classmethod
    @pytest.fixture()
    def table(cls):
        return cls.create(100, ['time', 'snr', 'frequency',
                                'duration', 'bandwidth'])


class TestEventTablePlot(EventTableMixin, TestPlot):
    def test_init_with_table(self, table):
        self.FIGURE_CLASS(table, 'time', 'frequency').close()
        with pytest.raises(ValueError):
            self.FIGURE_CLASS(table)
        self.FIGURE_CLASS(
            table, 'time', 'frequency', 'snr').close()
        self.FIGURE_CLASS(
            table, 'time', 'frequency', 'snr').close()


class TestEventTableAxes(EventTableMixin, TestAxes):
    def test_plot_table(self, table):
        fig, ax = self.new()
        snrs = table.get_column('snr')
        snrs.sort()
        # test with color
        c = ax.plot_table(table, 'time', 'frequency', 'snr')
        shape = c.get_offsets().shape
        assert isinstance(c, PathCollection)
        assert shape[0] == len(table)
        nptest.assert_array_equal(c.get_array(), snrs)
        # test with size_by
        with pytest.warns(DeprecationWarning):
            c = ax.plot_table(table, 'time', 'frequency', size_by='snr')
        # test with color and size_by
        with pytest.warns(DeprecationWarning):
            c = ax.plot_table(table, 'time', 'frequency', 'snr', size_by='snr')
        nptest.assert_array_equal(c.get_array(), snrs)

    def test_plot_tiles(self, table):
        fig, ax = self.new()
        snrs = table.get_column('snr')
        snrs.sort()
        # test with color
        c = ax.plot_tiles(table, 'time', 'frequency', 'duration',
                          'bandwidth', 'snr')
        assert isinstance(c, PolyCollection)
        # test other anchors
        c = ax.plot_tiles(table, 'time', 'frequency', 'duration',
                          'bandwidth', 'snr', anchor='ll')
        c = ax.plot_tiles(table, 'time', 'frequency', 'duration',
                          'bandwidth', 'snr', anchor='lr')
        c = ax.plot_tiles(table, 'time', 'frequency', 'duration',
                          'bandwidth', 'snr', anchor='ul')
        c = ax.plot_tiles(table, 'time', 'frequency', 'duration',
                          'bandwidth', 'snr', anchor='ur')
        with pytest.raises(ValueError):
            ax.plot_tiles(table, 'time', 'frequency', 'duration',
                          'bandwidth', 'snr', anchor='other')

    def test_get_column_string(self):
        with rc_context(rc={'text.usetex': True}):
            assert get_column_string('snr') == 'SNR'
            assert get_column_string('reduced_chisq') == r'Reduced $\chi^2$'
            assert get_column_string('flow') == r'f$_{\mbox{\small low}}$'
            assert get_column_string('end_time_ns') == r'End Time $(ns)$'

    def test_add_loudest(self, usetex, table):
        table.add_column(table.Column(data=['test'] * len(table), name='test'))
        loudest = table[table['snr'].argmax()]
        t, f, s = loudest['time'], loudest['frequency'], loudest['snr']

        # make plot
        fig, ax = self.new()
        ax.scatter(table['time'], table['frequency'])
        tpos = ax.title.get_position()

        # call function
        coll, text = ax.add_loudest(
            table, 'snr',  # table, rank
            'time', 'frequency',  # x, y
            'test',  # extra columns to print
            'time',  # duplicate (shouldn't get printed)
        )

        # check marker was placed at the right point
        utils.assert_array_equal(coll.get_offsets(), [(t, f)])

        # check text
        result = ('Loudest event: Time = {0}, Frequency = {1}, SNR = {2}, '
                  'Test = test'.format(
                      *('{0:.2f}'.format(x) for x in (t, f, s))))

        assert text.get_text() == result
        assert text.get_position() == (.5, 1.)

        # assert title got moved
        assert ax.title.get_position() == (tpos[0], tpos[1] + .05)

        # -- with more kwargs

        _, t = ax.add_loudest(table, 'snr', 'time', 'frequency',
                              position=(0., 0.), ha='left', va='top')
        assert t.get_position() == (0., 0.)

        # assert title doesn't get moved again if we specify position
        assert ax.title.get_position() == (tpos[0], tpos[1] + .05)

        # assert kw handling
        assert t.get_horizontalalignment() == 'left'
        assert t.get_verticalalignment() == 'top'

        self.save_and_close(fig)


# -- Segment plotter ----------------------------------------------------------

class SegmentMixin(object):
    FIGURE_CLASS = SegmentPlot
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


class TestSegmentPlot(SegmentMixin, TestPlot):
    def test_add_bitmask(self, segments):
        fig, ax = self.new()
        ax.plot(segments)
        maxes = fig.add_bitmask(0b0110)
        assert maxes.name == ax.name  # test same type
        maxes = fig.add_bitmask('0b0110', topdown=True)


class TestSegmentAxes(SegmentMixin, TestAxes):

    def test_plot_dqflag(self, flag):
        fig, ax = self.new()
        c = ax.plot_dqflag(flag)
        assert c.get_label() == flag.texname
        assert len(ax.collections) == 2
        assert ax.collections[1] is c

    def test_build_segment(self):
        patch = self.AXES_CLASS.build_segment((1.1, 2.4), 10)
        assert patch.get_xy(), (1.1, 9.6)
        assert numpy.isclose(patch.get_height(), 0.8)
        assert numpy.isclose(patch.get_width(), 1.3)
        assert patch.get_facecolor(), COLOR0
        # check kwarg passing
        patch = self.AXES_CLASS.build_segment((1.1, 2.4), 10, facecolor='red')
        assert patch.get_facecolor() == COLOR_CONVERTER.to_rgba('red')
        # check valign
        patch = self.AXES_CLASS.build_segment((1.1, 2.4), 10, valign='top')
        assert patch.get_xy() == (1.1, 9.2)
        patch = self.AXES_CLASS.build_segment((1.1, 2.4), 10, valign='bottom')
        assert patch.get_xy() == (1.1, 10.0)

    def test_plot_segmentlist(self, segments):
        fig, ax = self.new()
        c = ax.plot_segmentlist(segments)
        assert isinstance(c, PatchCollection)
        assert numpy.isclose(ax.dataLim.x0, 0.)
        assert numpy.isclose(ax.dataLim.x1, 7.)
        assert len(c.get_paths()) == len(segments)
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
        self.save_and_close(fig)

    def test_plot_segmentlistdict(self, segments):
        sld = SegmentListDict()
        sld['TEST'] = segments
        fig, ax = self.new()
        ax.plot(sld)
        self.save_and_close(fig)

    def test_plot(self, segments, flag):
        fig, ax = self.new()
        ax.plot(segments)
        ax.plot(flag)
        ax.plot(flag, segments)
        self.save_and_close(fig)

    def test_insetlabels(self, segments):
        fig, ax = self.new()
        ax.plot(segments)
        ax.set_insetlabels(True)
        self.save_and_close(fig)


# -- Histogram plotter --------------------------------------------------------

class HistogramMixin(object):
    FIGURE_CLASS = HistogramPlot
    AXES_CLASS = HistogramAxes

    @classmethod
    def setup_class(cls):
        numpy.random.seed(0)
        cls.ts = TimeSeries(numpy.random.rand(10000), sample_rate=128)


class TestHistogramPlot(HistogramMixin, TestPlot):
    def test_init(self):
        super(TestHistogramPlot, self).test_init()

        # test with data
        bins = [1, 2, 5]
        plot = self.FIGURE_CLASS([1, 2, 3, 4, 5], bins=bins)
        assert len(plot.axes) == 1
        ax = plot.gca()
        assert len(ax.containers) == 1
        assert len(ax.containers[0].patches) == len(bins) - 1


class TestHistogramAxes(HistogramMixin, TestAxes):
    def test_hist_series(self):
        fig, ax = self.new()
        ax.hist_series(self.ts)

    def test_hist(self):
        fig, ax = self.new()
        ax.hist(self.ts)

    @utils.skip_missing_dependency('glue')
    def test_common_limits(self):
        fig, ax = self.new()
        a = ax.common_limits(self.ts.value)
        assert numpy.allclose(a, (7.2449638492178003e-05, 0.9999779517807228))
        b = ax.common_limits([self.ts.value])
        assert a == b

    def test_bin_boundaries(self):
        ax = self.AXES_CLASS
        nptest.assert_array_equal(ax.bin_boundaries(0, 10, 4),
                                  [0., 2.5, 5., 7.5, 10.])
        nptest.assert_array_equal(ax.bin_boundaries(1, 100, 2, log=True),
                                  [1, 10., 100.])
        with pytest.raises(ValueError):
            ax.bin_boundaries(0, 100, 2, log=True)

    def test_histogram_weights(self):
        fig, ax = self.new()
        ax.hist(numpy.random.random(1000), weights=10.)
        self.save_and_close(fig)


# -- Filter plotter -----------------------------------------------------------

class TestBodePlot(PlottingTestBase):
    FIGURE_CLASS = BodePlot

    def test_init(self):
        fig = self.FIGURE_CLASS()
        assert len(fig.axes) == 2
        maxes, paxes = fig.axes
        # test magnigtude axes
        assert isinstance(maxes, FrequencySeriesAxes)
        assert maxes.get_xscale() == 'log'
        assert maxes.get_xlabel() == ''
        assert maxes.get_yscale() == 'linear'
        assert maxes.get_ylabel() == 'Magnitude [dB]'
        # test phase axes
        assert isinstance(paxes, FrequencySeriesAxes)
        assert paxes.get_xscale() == 'log'
        assert paxes.get_xlabel() == 'Frequency [Hz]'
        assert paxes.get_yscale() == 'linear'
        assert paxes.get_ylabel() == 'Phase [deg]'

    def test_add_filter(self):
        # test method 1
        fig = self.FIGURE_CLASS()
        lm, lp = fig.add_filter(ZPK, analog=True)
        assert lm is fig.maxes.get_lines()[-1]
        assert lp is fig.paxes.get_lines()[-1]
        nptest.assert_array_equal(lm.get_xdata(), FREQUENCIES)
        nptest.assert_array_equal(lm.get_ydata(), MAGNITUDE)
        nptest.assert_array_equal(lp.get_xdata(), FREQUENCIES)
        nptest.assert_array_almost_equal(lp.get_ydata(), PHASE)
        self.save_and_close(fig)
        # test method 2
        fig = self.FIGURE_CLASS(ZPK, analog=True)
        lm = fig.maxes.get_lines()[0]
        lp = fig.paxes.get_lines()[0]
        nptest.assert_array_equal(lm.get_xdata(), FREQUENCIES)
        nptest.assert_array_equal(lm.get_ydata(), MAGNITUDE)
        nptest.assert_array_equal(lp.get_xdata(), FREQUENCIES)
        nptest.assert_array_almost_equal(lp.get_ydata(), PHASE)


# -- gwpy.plotter.gps module tests --------------------------------------------

class TestGpsTransform(object):
    TRANSFORM = GPSTransform
    EPOCH = 100.0
    UNIT = 'minutes'
    SCALE = 60.
    X = 190.0
    A = 90.0
    B = 19/6.
    C = 1.5

    def test_empty(self):
        t = self.TRANSFORM()
        assert t.transform(1.0) == 1.0

    def test_epoch(self):
        transform = self.TRANSFORM(epoch=self.EPOCH)
        assert transform.get_epoch() == self.EPOCH
        assert transform.transform(self.X) == self.A
        assert numpy.isclose(
            transform.inverted().transform(transform.transform(self.X)),
            self.X)

    def test_scale(self):
        transform = self.TRANSFORM(unit=self.UNIT)
        assert transform.get_scale() == self.SCALE
        assert transform.transform(self.X) == self.B
        assert numpy.isclose(
            transform.inverted().transform(transform.transform(self.X)),
            self.X)

    def test_epoch_and_scale(self):
        transform = self.TRANSFORM(epoch=self.EPOCH, unit=self.UNIT)
        assert transform.transform(self.X) == self.C
        assert numpy.isclose(
            transform.inverted().transform(transform.transform(self.X)),
            self.X)


class TestInverseGpsTransform(TestGpsTransform):
    TRANSFORM = InvertedGPSTransform
    A = 290.0
    B = 11400.0
    C = 11500.0


# -- gwpy.plotter.text module tests -------------------------------------------

class TestText(PlottingTestBase):
    def test_to_string(self):
        # test without latex
        with rc_context(rc={'text.usetex': False}):
            assert to_string('test') == 'test'
            assert to_string(4.0) == '4.0'
            assert to_string(8) == '8'
        with rc_context(rc={'text.usetex': True}):
            assert to_string('test') == 'test'
            assert to_string(2000) == r'2\!\!\times\!\!10^{3}'
            assert to_string(8) == '8'

    def test_unit_as_label(self):
        # just test basics, latex formatting is tested elsewhere
        with rc_context(rc={'text.usetex': False}):
            assert unit_as_label(units.Hz) == 'Frequency [Hz]'
            assert unit_as_label(units.Volt) == 'Electrical Potential [V]'


# -- gwpy.plotter.tex module tests --------------------------------------------

class TestTex(PlottingTestBase):
    def test_float_to_latex(self):
        assert float_to_latex(1) == '1'
        assert float_to_latex(100) == '10^{2}'
        assert float_to_latex(-500) == r'-5\!\!\times\!\!10^{2}'
        with pytest.raises(TypeError):
            float_to_latex('1')

    def test_label_to_latex(self):
        assert label_to_latex(None) == ''
        assert label_to_latex('') == ''
        assert label_to_latex('Test') == 'Test'
        assert label_to_latex('Test_with_underscore') == (
            r'Test\_with\_underscore')
        assert label_to_latex(r'Test_with\_escaped\%characters') == (
            r'Test\_with\_escaped\%characters')

    def test_unit_to_latex(self):
        t = unit_to_latex(units.Hertz)
        assert t == r'$\mathrm{Hz}$'
        t = unit_to_latex(units.Volt.decompose())
        assert t == r'$\mathrm{m^{2}\,kg\,A^{-1}\,s^{-3}}$'


# -- gwpy.plotter.html module tests -------------------------------------------

class TestHtml(object):
    def test_map_data(self):
        numpy.random.seed(0)
        data = numpy.vstack((numpy.arange(100), numpy.random.random(100)))
        fig = figure()
        ax = fig.gca()
        html = map_data(data, ax, 'test.png')
        assert html.startswith('<!doctype html>')
        html = map_data(data, ax, 'test.png', standalone=False)
        assert html.startswith('\n<img src="test.png"')

    @utils.skip_missing_dependency('bs4')
    def test_map_artist(self):
        from bs4 import BeautifulSoup

        # create figure and plot a line
        fig = figure()
        ax = fig.gca()
        line = ax.plot([1, 2, 3, 4, 5])[0]
        data = list(zip(*line.get_data()))

        # create HTML map
        html = map_artist(line, 'test.png')

        # validate HTML map
        soup = BeautifulSoup(html, 'html.parser')
        areas = soup.find_all('area')
        assert len(areas) == 5
        assert sorted([eval(a.attrs['alt']) for a in areas]) == data
