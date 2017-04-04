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

import os
import tempfile
from math import pi

import numpy
from numpy import testing as nptest

from scipy import signal

from matplotlib import (use, rc_context, __version__ as mpl_version)
use('agg')
from matplotlib.legend import Legend
from matplotlib.colors import (LogNorm, ColorConverter)
from matplotlib.collections import (PathCollection, PatchCollection,
                                    PolyCollection)

from astropy import units

from compat import unittest

from gwpy.segments import (DataQualityFlag,
                           Segment, SegmentList, SegmentListDict)
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from gwpy.table import EventTable
from gwpy.plotter import (figure, rcParams, Plot, Axes,
                          TimeSeriesPlot, TimeSeriesAxes,
                          FrequencySeriesPlot, FrequencySeriesAxes,
                          EventTablePlot, EventTableAxes,
                          HistogramPlot, HistogramAxes,
                          SegmentPlot, SegmentAxes,
                          SpectrogramPlot, BodePlot)
from gwpy.plotter import utils
from gwpy.plotter.gps import (GPSTransform, InvertedGPSTransform)
from gwpy.plotter.html import map_data
from gwpy.plotter.tex import (float_to_latex, label_to_latex,
                              unit_to_latex, USE_TEX)
from gwpy.plotter.table import get_column_string

from test_timeseries import TEST_HDF_FILE
from test_table import TEST_OMEGA_FILE

# design ZPK for BodePlot test
ZPK = [100], [1], 1e-2
FREQUENCIES, H = signal.freqresp(ZPK, n=100)
MAGNITUDE = 20 * numpy.log10(numpy.absolute(H))
PHASE = numpy.degrees(numpy.unwrap(numpy.angle(H)))

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


class Mixin(object):
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
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            fig.save(f.name)
        return fig

    def save_and_close(self, fig, suffix='.png'):
        self.save(fig, suffix=suffix)
        fig.close()
        return fig


class PlotTestCase(Mixin, unittest.TestCase):
    """`TestCase` for the `gwpy.plotter` module
    """
    def test_init(self):
        # test object creation
        fig, ax = self.new()
        self.assertIsInstance(fig, self.FIGURE_CLASS)
        # test added properties
        self.assertIsInstance(fig.colorbars, list)
        self.assertEqual(len(fig.colorbars), 0)
        self.assertIsInstance(fig._coloraxes, list)
        self.assertEqual(len(fig._coloraxes), 0)
        # test gca
        self.assertIsInstance(ax, self.AXES_CLASS)
        self.save_and_close(fig)

    def test_auto_refresh(self):
        fig = self.FIGURE_CLASS()
        self.assertFalse(fig.get_auto_refresh())
        fig.set_auto_refresh(True)
        self.assertTrue(fig.get_auto_refresh())
        fig = self.FIGURE_CLASS(auto_refresh=True)
        self.assertTrue(fig.get_auto_refresh())
        self.save_and_close(fig)

    def test_subplotpars(self):
        # check that dynamic subplotpars gets applied
        fig, ax = self.new(figsize=(12, 4))
        target = utils.SUBPLOT_WIDTH[12] + utils.SUBPLOT_HEIGHT[4]
        sbp = fig.subplotpars
        self.assertTupleEqual(target,
                              (sbp.left, sbp.right, sbp.bottom, sbp.top))
        # check that dynamic subplotpars doesn't get applied if the user
        # overrides any of the settings
        with rc_context(rc={'figure.subplot.left': target[0]*.1}):
            fig, ax = self.new(figsize=(12, 4))
            sbp = fig.subplotpars
            self.assertEqual(sbp.left, target[0]*.1)

    # -- test axes_method decorators
    def test_axes_methods(self):
        fig, ax = self.new()
        # get_xlim
        self.assertTupleEqual(fig.get_xlim(), ax.get_xlim())
        # set_xlim
        fig.set_xlim(4, 5)
        self.assertTupleEqual(fig.get_xlim(), (4, 5))
        self.assertTupleEqual(fig.get_xlim(), ax.get_xlim())

        # get_xlabel
        self.assertEqual(fig.get_xlabel(), ax.get_xlabel())
        # set_xlabel
        fig.set_xlabel('test')
        self.assertEqual(fig.get_xlabel(), 'test')
        self.assertEqual(fig.get_xlabel(), ax.get_xlabel())

        # get_ylim
        self.assertTupleEqual(fig.get_ylim(), ax.get_ylim())
        # set_ylim
        fig.set_ylim(4, 5)
        self.assertTupleEqual(fig.get_ylim(), (4, 5))
        self.assertTupleEqual(fig.get_ylim(), ax.get_ylim())

        # get_ylabel
        self.assertEqual(fig.get_ylabel(), ax.get_ylabel())
        # set_ylabel
        fig.set_ylabel('test')
        self.assertEqual(fig.get_ylabel(), 'test')
        self.assertEqual(fig.get_ylabel(), ax.get_ylabel())

        # title
        self.assertEqual(fig.get_title(), ax.get_title())
        # set_title
        fig.set_title('Test')
        self.assertEqual(fig.get_title(), 'Test')
        self.assertEqual(fig.get_title(), ax.get_title())

        # xscale
        self.assertEqual(fig.get_xscale(), ax.get_xscale())
        # set_xscale
        fig.set_xscale('log')
        self.assertEqual(fig.get_xscale(), 'log')
        self.assertEqual(fig.get_xscale(), ax.get_xscale())

        # yscale
        self.assertEqual(fig.get_yscale(), ax.get_yscale())
        # set_yscale
        fig.set_yscale('log')
        self.assertEqual(fig.get_yscale(), 'log')
        self.assertEqual(fig.get_yscale(), ax.get_yscale())
        self.save_and_close(fig)

    def test_logx(self):
        fig, ax = self.new()
        fig.set_xlim(0.1, 10)
        fig.logx = True
        self.assertEqual(ax.get_xscale(), 'log')
        self.assertTrue(fig.logx)
        self.save_and_close(fig)

    def test_logy(self):
        fig, ax = self.new()
        fig.set_ylim(0.1, 10)
        fig.logy = True
        self.assertEqual(ax.get_yscale(), 'log')
        self.assertTrue(fig.logy)
        self.save_and_close(fig)

    def test_add_legend(self):
        fig, ax = self.new()
        self.assertIsNone(fig.add_legend())
        ax.plot([1, 2, 3, 4], label='Plot')
        self.assertIsInstance(fig.add_legend(), Legend)
        self.save_and_close(fig)

    def test_figure(self):
        fig = figure()
        self.assertIsInstance(fig, Plot)
        fig.close()
        fig = figure(FigureClass=self.FIGURE_CLASS)
        self.assertIsInstance(fig, self.FIGURE_CLASS)
        fig.close()

    def test_add_line(self):
        fig, ax = self.new()
        fig.add_line([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertEqual(len(ax.lines), 1)
        fig.close()

    def test_add_scatter(self):
        fig, ax = self.new()
        fig.add_scatter([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertEqual(len(ax.collections), 1)
        fig.close()

    def test_add_arrays(self):
        ts = TimeSeries([1, 2, 3, 4])
        fs = FrequencySeries([1, 2, 3, 4])
        fig = self.FIGURE_CLASS()
        self.assertEqual(len(fig.axes), 0)
        fig.add_timeseries(ts)
        self.assertEqual(len(fig.axes), 1)
        self.assertIsInstance(fig.axes[0], TimeSeriesAxes)
        fig.add_frequencyseries(fs)
        self.assertEqual(len(fig.axes), 2)
        self.assertIsInstance(fig.axes[1], FrequencySeriesAxes)
        fig.close()


class AxesTestCase(Mixin, unittest.TestCase):
    def test_legend(self):
        fig = self.FIGURE_CLASS()
        ax = fig.add_subplot(111, projection=self.AXES_CLASS.name)
        ax.plot([1, 2, 3, 4], label='Plot')
        leg = ax.legend()
        self.assertEqual(leg.get_frame().get_alpha(), 0.8)
        for l in leg.get_lines():
            self.assertEqual(l.get_linewidth(), 8)
        self.save_and_close(fig)

    def test_matplotlib_compat(self):
        for method in ['plot', 'loglog', 'semilogx', 'semilogy']:
            fig, ax = self.new()
            plot = getattr(ax, method)
            print(type(fig), type(ax), plot)
            plot([1, 2, 3, 4, 5])
            plot([1, 2, 3, 4, 5], color='green', linestyle='--')
            self.save_and_close(fig)


# -- TimeSeries plotters ------------------------------------------------------

class TimeSeriesMixin(object):
    FIGURE_CLASS = TimeSeriesPlot
    AXES_CLASS = TimeSeriesAxes

    def setUp(self):
        self.ts = TimeSeries.read(TEST_HDF_FILE, 'H1:LDAS-STRAIN')
        self.sg = self.ts.spectrogram2(0.5, 0.49)
        self.mmm = [self.ts, self.ts * 0.9, self.ts*1.1]


class TimeSeriesPlotTestCase(TimeSeriesMixin, PlotTestCase):
    def test_init(self):
        # test empty
        fig, ax = self.new()
        self.assertIsInstance(ax, self.AXES_CLASS)
        # test passing arguments
        fig = self.FIGURE_CLASS(figsize=[9, 6])
        self.assertEqual(fig.get_figwidth(), 9)
        fig = self.FIGURE_CLASS(self.ts)
        ax = fig.gca()
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(fig.get_epoch(), self.ts.x0.value)
        self.assertEqual(fig.get_xlim(), self.ts.span)
        # test passing multiple timeseries
        fig = self.FIGURE_CLASS(self.ts, self.ts)
        self.assertEqual(len(fig.gca().lines), 2)
        fig = self.FIGURE_CLASS(self.ts, self.ts, sep=True, sharex=True)
        self.assertEqual(len(fig.axes), 2)
        for ax in fig.axes:
            self.assertEqual(len(ax.lines), 1)
        self.assertIs(fig.axes[1]._sharex, fig.axes[0])
        # test kwarg parsing
        fig = self.FIGURE_CLASS(self.ts, figsize=[12, 6], rasterized=True)

    def test_add_colorbar(self):
        def make_fig():
            fig = self.FIGURE_CLASS(self.sg ** (1/2.))
            return fig, fig.gca()

        # test basic
        fig, ax = make_fig()
        cb = fig.add_colorbar()
        self.assertEqual(len(fig.colorbars), 1)
        self.assertIs(cb, fig.colorbars[0])
        self.save_and_close(fig)
        # test kwargs
        fig, ax = make_fig()
        fig.add_colorbar(ax=ax, norm='log', clim=[1e-22, 1e-18],
                         label='Test colorbar')
        self.save_and_close(fig)


class TimeSeriesAxesTestCase(TimeSeriesMixin, AxesTestCase):
    def test_init(self):
        fig, ax = self.new()
        self.assertIsInstance(ax, self.AXES_CLASS)
        self.assertEqual(ax.get_epoch(), 0)
        self.assertEqual(ax.get_xscale(), 'auto-gps')
        self.assertEqual(ax.get_xlabel(), '_auto')
        self.save_and_close(fig)

    def test_plot_timeseries(self):
        fig, ax = self.new()
        # check method works
        ax.plot_timeseries(self.ts)
        # check data are correct
        l = ax.get_lines()[0]
        nptest.assert_array_equal(l.get_xdata(), self.ts.times.value)
        nptest.assert_array_equal(l.get_ydata(), self.ts.value)
        # check GPS axis is set ok
        self.assertEqual(ax.get_epoch(), self.ts.x0.value)
        self.assertTupleEqual(ax.get_xlim(), tuple(self.ts.span))
        self.save_and_close(fig)

    def test_plot_timeseries_mmm(self):
        fig, ax = self.new()
        # test default
        artists = ax.plot_timeseries_mmm(*self.mmm)
        self.assertEqual(len(artists), 5)
        self.assertEqual(len(ax.lines), 3)
        self.assertEqual(len(ax.collections), 2)
        self.save_and_close(fig)
        # test min only
        fig, ax = self.new()
        artists = ax.plot_timeseries_mmm(self.mmm[0], min_=self.mmm[1])
        self.assertEqual(len(artists), 5)
        self.assertIsNone(artists[3])
        self.assertIsNone(artists[4])
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(len(ax.collections), 1)
        self.save_and_close(fig)
        # test max only
        fig, ax = self.new()
        artists = ax.plot_timeseries_mmm(self.mmm[0], max_=self.mmm[2])
        self.assertEqual(len(artists), 5)
        self.assertIsNone(artists[1])
        self.assertIsNone(artists[2])
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(len(ax.collections), 1)

    def test_plot_spectrogram(self):
        fig, ax = self.new()
        # check method
        ax.plot_spectrogram(self.sg)
        coll = ax.collections[0]
        nptest.assert_array_equal(coll.get_array(), self.sg.value.T.flatten())
        # check GPS axis is set ok
        self.assertEqual(ax.get_epoch(), self.sg.x0.value)
        self.assertTupleEqual(ax.get_xlim(), tuple(self.sg.xspan))
        # check frequency axis
        if self.use_tex:
            self.assertEqual(ax.get_ylabel(), r'Frequency [$\mathrm{Hz}$]')
        else:
            self.assertEqual(ax.get_ylabel(), r'Frequency [Hz]')
        # check kwarg parsing
        c = ax.plot_spectrogram(self.sg, norm='log')
        self.assertIsInstance(c.norm, LogNorm)
        self.save_and_close(fig)

    def test_plot(self):
        fig, ax = self.new()
        ax.plot(self.ts)
        self.save_and_close(fig)


# -- FrequencySeries plotters -------------------------------------------------

class FrequencySeriesMixin(object):
    FIGURE_CLASS = FrequencySeriesPlot
    AXES_CLASS = FrequencySeriesAxes

    def setUp(self):
        self.ts = TimeSeries.read(TEST_HDF_FILE, 'H1:LDAS-STRAIN')
        self.asd = self.ts.asd(1)
        self.mmm = [self.asd, self.asd*0.9, self.asd*1.1]

class FrequencySeriesPlotTestCase(FrequencySeriesMixin, PlotTestCase):
    def test_init(self):
        super(FrequencySeriesPlotTestCase, self).test_init()
        # test convenience plotting
        fig = self.FIGURE_CLASS(self.asd)
        self.assertEqual(len(fig.axes), 1)
        ax = fig.gca()
        self.assertIsInstance(ax, self.AXES_CLASS)
        self.assertEqual(len(ax.lines), 1)


class FrequencySeriesAxesTestCase(FrequencySeriesMixin, AxesTestCase):
    def test_plot_frequencyseries(self):
        fig, ax = self.new()
        l = ax.plot_frequencyseries(self.asd)[0]
        nptest.assert_array_equal(l.get_xdata(), self.asd.frequencies.value)
        nptest.assert_array_equal(l.get_ydata(), self.asd.value)
        self.save_and_close(fig)

    def test_plot_frequencyseries_mmm(self):
        fig, ax = self.new()
        # test defaults
        artists = ax.plot_frequencyseries_mmm(*self.mmm)
        self.assertEqual(len(artists), 5)
        self.assertEqual(len(ax.lines), 3)
        self.assertEqual(len(ax.collections), 2)
        self.save_and_close(fig)
        # test min only
        fig, ax = self.new()
        artists = ax.plot_frequencyseries_mmm(self.mmm[0], min_=self.mmm[1])
        self.assertEqual(len(artists), 5)
        self.assertIsNone(artists[3])
        self.assertIsNone(artists[4])
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(len(ax.collections), 1)
        self.save_and_close(fig)
        # test max only
        fig, ax = self.new()
        artists = ax.plot_frequencyseries_mmm(self.mmm[0], max_=self.mmm[2])
        self.assertEqual(len(artists), 5)
        self.assertIsNone(artists[1])
        self.assertIsNone(artists[2])
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(len(ax.collections), 1)


# -- Table plotters -----------------------------------------------------------

class EventTableMixin(object):
    FIGURE_CLASS = EventTablePlot
    AXES_CLASS = EventTableAxes

    def setUp(self):
        self.table = EventTable.read(TEST_OMEGA_FILE, format='ascii.omega')


class EventTablePlotTestCase(EventTableMixin, PlotTestCase):
    def test_init_with_table(self):
        self.FIGURE_CLASS(self.table, 'time', 'frequency').close()
        self.assertRaises(ValueError, self.FIGURE_CLASS, self.table)
        self.FIGURE_CLASS(
            self.table, 'time', 'frequency', 'normalizedEnergy').close()
        self.FIGURE_CLASS(
            self.table, 'time', 'frequency', 'normalizedEnergy').close()


class EventTableAxesTestCase(EventTableMixin, AxesTestCase):
    def test_plot_table(self):
        fig, ax = self.new()
        snrs = self.table.get_column('normalizedEnergy')
        snrs.sort()
        # test with color
        c = ax.plot_table(self.table, 'time', 'frequency', 'normalizedEnergy')
        shape = c.get_offsets().shape
        self.assertIsInstance(c, PathCollection)
        self.assertEqual(shape[0], len(self.table))
        nptest.assert_array_equal(c.get_array(), snrs)
        # test with size_by
        c = ax.plot_table(self.table, 'time', 'frequency',
                          size_by='normalizedEnergy')
        # test with color and size_by
        c = ax.plot_table(self.table, 'time', 'frequency', 'normalizedEnergy',
                          size_by='normalizedEnergy')
        nptest.assert_array_equal(c.get_array(), snrs)
        # test add_loudest
        ax.set_title('title')
        ax.add_loudest(self.table, 'normalizedEnergy', 'time', 'frequency')

    def test_plot_tiles(self):
        fig, ax = self.new()
        snrs = self.table.get_column('normalizedEnergy')
        snrs.sort()
        # test with color
        c = ax.plot_tiles(self.table, 'time', 'frequency', 'duration',
                          'bandwidth', 'normalizedEnergy')
        shape = c.get_offsets().shape
        self.assertIsInstance(c, PolyCollection)
        # test other anchors
        c = ax.plot_tiles(self.table, 'time', 'frequency', 'duration',
                          'bandwidth', 'normalizedEnergy', anchor='ll')
        c = ax.plot_tiles(self.table, 'time', 'frequency', 'duration',
                          'bandwidth', 'normalizedEnergy', anchor='lr')
        c = ax.plot_tiles(self.table, 'time', 'frequency', 'duration',
                          'bandwidth', 'normalizedEnergy', anchor='ul')
        c = ax.plot_tiles(self.table, 'time', 'frequency', 'duration',
                          'bandwidth', 'normalizedEnergy', anchor='ur')
        self.assertRaises(ValueError, ax.plot_tiles, self.table, 'time',
                          'frequency', 'duration', 'bandwidth',
                          'normalizedEnergy', anchor='other')

    def test_get_column_string(self):
        rcParams['text.usetex'] = True
        self.assertEqual(get_column_string('snr'), 'SNR')
        self.assertEqual(get_column_string('reduced_chisq'),
                         r'Reduced $\chi^2$')
        self.assertEqual(get_column_string('flow'),
                         r'f$_{\mbox{\small low}}$')
        self.assertEqual(get_column_string('end_time_ns'),
                         r'End Time $(ns)$')


# -- Segment plotter ----------------------------------------------------------

class SegmentMixin(object):
    FIGURE_CLASS = SegmentPlot
    AXES_CLASS = SegmentAxes

    def setUp(self):
        self.segments = SegmentList([Segment(0, 3), Segment(6, 7)])
        active = SegmentList([Segment(1, 2), Segment(3, 4), Segment(5, 7)])
        self.flag = DataQualityFlag(name='Test segments', known=self.segments,
                                    active=active)


class SegmentPlotTestCase(SegmentMixin, PlotTestCase):
    def test_add_bitmask(self):
        fig, ax = self.new()
        ax.plot(self.segments)
        maxes = fig.add_bitmask(0b0110)
        self.assertEqual(maxes.name, ax.name)  # test same type
        maxes = fig.add_bitmask('0b0110', topdown=True)


class SegmentAxesTestCase(SegmentMixin, AxesTestCase):

    def test_plot_dqflag(self):
        fig, ax = self.new()
        c = ax.plot_dqflag(self.flag)
        self.assertEqual(c.get_label(), self.flag.texname)
        self.assertEqual(len(ax.collections), 2)
        self.assertIs(ax.collections[1], c)

    def test_build_segment(self):
        patch = self.AXES_CLASS.build_segment((1.1, 2.4), 10)
        self.assertTupleEqual(patch.get_xy(), (1.1, 9.6))
        self.assertAlmostEqual(patch.get_height(), 0.8)
        self.assertAlmostEqual(patch.get_width(), 1.3)
        self.assertTupleEqual(patch.get_facecolor(), COLOR0)
        # check kwarg passing
        patch = self.AXES_CLASS.build_segment((1.1, 2.4), 10, facecolor='red')
        self.assertTupleEqual(patch.get_facecolor(),
                              COLOR_CONVERTER.to_rgba('red'))
        # check valign
        patch = self.AXES_CLASS.build_segment((1.1, 2.4), 10, valign='top')
        self.assertTupleEqual(patch.get_xy(), (1.1, 9.2))
        patch = self.AXES_CLASS.build_segment((1.1, 2.4), 10, valign='bottom')
        self.assertTupleEqual(patch.get_xy(), (1.1, 10.0))

    def test_plot_segmentlist(self):
        fig, ax = self.new()
        c = ax.plot_segmentlist(self.segments)
        self.assertIsInstance(c, PatchCollection)
        self.assertAlmostEqual(ax.dataLim.x0, 0.)
        self.assertAlmostEqual(ax.dataLim.x1, 7.)
        self.assertEqual(len(c.get_paths()), len(self.segments))
        self.assertEqual(ax.get_epoch(), self.segments[0][0])
        # test y
        p = ax.plot_segmentlist(self.segments).get_paths()[0].get_extents()
        self.assertEqual(p.y0 + p.height/2., 1.)
        p = ax.plot_segmentlist(
            self.segments, y=8).get_paths()[0].get_extents()
        self.assertEqual(p.y0 + p.height/2., 8.)
        # test kwargs
        c = ax.plot_segmentlist(self.segments, label='My segments',
                                rasterized=True)
        self.assertEqual(c.get_label(), 'My segments')
        self.assertTrue(c.get_rasterized())
        # test collection=False
        c = ax.plot_segmentlist(self.segments, collection=False, label='test')
        self.assertIsInstance(c, list)
        self.assertNotIsInstance(c, PatchCollection)
        self.assertEqual(c[0].get_label(), 'test')
        self.assertEqual(c[1].get_label(), '')
        self.assertEqual(len(ax.patches), len(self.segments))
        # test empty
        c = ax.plot_segmentlist(type(self.segments)())
        self.save_and_close(fig)

    def test_plot_segmentlistdict(self):
        sld = SegmentListDict()
        sld['TEST'] = self.segments
        fig, ax = self.new()
        ax.plot(sld)
        self.save_and_close(fig)

    def test_plot(self):
        fig, ax = self.new()
        ax.plot(self.segments)
        ax.plot(self.flag)
        ax.plot(self.flag, self.segments)
        self.save_and_close(fig)

    def test_insetlabels(self):
        fig, ax = self.new()
        ax.plot(self.segments)
        ax.set_insetlabels(True)
        self.save_and_close(fig)


# -- Histogram plotter --------------------------------------------------------

class HistogramMixin(object):
    FIGURE_CLASS = HistogramPlot
    AXES_CLASS = HistogramAxes

    def setUp(self):
        self.ts = TimeSeries.read(TEST_HDF_FILE, 'H1:LDAS-STRAIN')


class HistogramPlotTestCase(HistogramMixin, PlotTestCase):
    pass


class HistogramAxesTestCase(HistogramMixin, AxesTestCase):
    def test_hist_series(self):
        fig, ax = self.new()
        hist = ax.hist_series(self.ts)

    def test_hist(self):
        fig, ax = self.new()
        ax.hist(self.ts)

    def test_common_limits(self):
        fig, ax = self.new()
        a = ax.common_limits(self.ts.value * 1e16)
        self.assertTupleEqual(a, (-1.0227435293, 1.0975343221))
        b = ax.common_limits([self.ts.value * 1e16])
        self.assertEqual(a, b)

    def test_bin_boundaries(self):
        ax = self.AXES_CLASS
        nptest.assert_array_equal(ax.bin_boundaries(0, 10, 4),
                                  [0., 2.5, 5., 7.5, 10.])
        nptest.assert_array_equal(ax.bin_boundaries(1, 100, 2, log=True),
                                  [1, 10., 100.])
        self.assertRaises(ValueError, ax.bin_boundaries, 0, 100, 2, log=True)

    def test_histogram_weights(self):
        fig, ax = self.new()
        ax.hist(numpy.random.random(1000), weights=10.)
        self.save_and_close(fig)


# -- Filter plotter -----------------------------------------------------------

class BodePlotTestCase(Mixin, unittest.TestCase):
    FIGURE_CLASS = BodePlot

    def test_init(self):
        fig = self.FIGURE_CLASS()
        self.assertEqual(len(fig.axes), 2)
        maxes, paxes = fig.axes
        # test magnigtude axes
        self.assertIsInstance(maxes, FrequencySeriesAxes)
        self.assertEqual(maxes.get_xscale(), 'log')
        self.assertEqual(maxes.get_xlabel(), '')
        self.assertEqual(maxes.get_yscale(), 'linear')
        self.assertEqual(maxes.get_ylabel(), 'Magnitude [dB]')
        # test phase axes
        self.assertIsInstance(paxes, FrequencySeriesAxes)
        self.assertEqual(paxes.get_xscale(), 'log')
        self.assertEqual(paxes.get_xlabel(), 'Frequency [Hz]')
        self.assertEqual(paxes.get_yscale(), 'linear')
        self.assertEqual(paxes.get_ylabel(), 'Phase [deg]')
        self.assertTupleEqual(paxes.get_ylim(), (-185, 185))

    def test_add_filter(self):
        # test method 1
        fig = self.FIGURE_CLASS()
        fig.add_filter(ZPK, analog=True)
        lm = fig.maxes.get_lines()[0]
        lp = fig.paxes.get_lines()[0]
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

class GpsTransformTestCase(unittest.TestCase):
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
        self.assertEqual(t.transform(1.0), 1.0)

    def test_epoch(self):
        transform = self.TRANSFORM(epoch=self.EPOCH)
        self.assertEqual(transform.get_epoch(), self.EPOCH)
        self.assertEqual(transform.transform(self.X), self.A)
        self.assertAlmostEqual(
            transform.inverted().transform(transform.transform(self.X)),
            self.X)

    def test_scale(self):
        transform = self.TRANSFORM(unit=self.UNIT)
        self.assertEqual(transform.get_scale(), self.SCALE)
        self.assertEqual(transform.transform(self.X), self.B)
        self.assertAlmostEqual(
            transform.inverted().transform(transform.transform(self.X)),
            self.X)

    def test_epoch_and_scale(self):
        transform = self.TRANSFORM(epoch=self.EPOCH, unit=self.UNIT)
        self.assertEqual(transform.transform(self.X), self.C)
        self.assertAlmostEqual(
            transform.inverted().transform(transform.transform(self.X)),
            self.X)


class InverseGpsTransformTestCase(GpsTransformTestCase):
    TRANSFORM = InvertedGPSTransform
    A = 290.0
    B = 11400.0
    C = 11500.0


# -- gwpy.plotter.tex module tests --------------------------------------------

class TexTestCase(Mixin, unittest.TestCase):
    def test_float_to_latex(self):
        self.assertEqual(float_to_latex(1), '1')
        self.assertEqual(float_to_latex(100), '10^{2}')
        self.assertEqual(float_to_latex(-500), r'-5\!\!\times\!\!10^{2}')
        self.assertRaises(TypeError, float_to_latex, '1')

    def test_label_to_latex(self):
        self.assertEqual(label_to_latex('Test'), 'Test')
        self.assertEqual(label_to_latex('Test_with_underscore'),
                         r'Test\_with\_underscore')

    def test_unit_to_latex(self):
        t = unit_to_latex(units.Hertz)
        self.assertEqual(t, r'$\mathrm{Hz}$')
        t = unit_to_latex(units.Volt.decompose())
        self.assertEqual(t, r'$\mathrm{m^{2}\,kg\,A^{-1}\,s^{-3}}$')


# -- gwpy.plotter.html module tests -------------------------------------------

class HtmlTestCase(unittest.TestCase):
    def setUp(self):
        self.data = numpy.vstack((numpy.arange(100), numpy.random.random(100)))

    def test_map_data(self):
        fig = figure()
        ax = fig.gca()
        html = map_data(self.data, ax, 'test.png')
        self.assertTrue(html.startswith('<!doctype html>'))
        html = map_data(self.data, ax, 'test.png', standalone=False)
        self.assertTrue(html.startswith('\n<img src="test.png"'))


if __name__ == '__main__':
    unittest.main()
