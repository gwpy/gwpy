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

"""An extension of the Plot class for handling TimeSeries
"""

import re
import datetime
import numpy
import itertools
import copy

from matplotlib import (pyplot, axes, cm, colors)
from matplotlib.projections import register_projection

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from lal import LIGOTimeGPS

from .core import Plot
from ..segments import SegmentList
from ..time import Time
from . import (ticks, tex)
from .axes import Axes
from .decorators import auto_refresh

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['TimeSeriesPlot', 'TimeSeriesAxes']


class TimeSeriesPlot(Plot):
    """An extension of the :class:`~gwpy.plotter.core.Plot` class for
    displaying data from :class:`~gwpy.timeseries.core.TimeSeries`

    Parameters
    ----------
    *series : `TimeSeries`
        any number of :class:`~gwpy.timeseries.core.TimeSeries` to
        display on the plot
    **kwargs
        other keyword arguments as applicable for the
        :class:`~gwpy.plotter.core.Plot`
    """
    def __new__(cls, *series, **kwargs):
        kwargs.setdefault('figsize', [12, 6])
        return super(TimeSeriesPlot, cls).__new__(cls, **kwargs)

    def __init__(self, *series, **kwargs):
        """Initialise a new TimeSeriesPlot
        """
        sep = kwargs.pop('sep', False)

        # generate figure
        super(TimeSeriesPlot, self).__init__(**kwargs)

        # plot data
        for ts in series:
            self.add_timeseries(ts, newax=sep)

        # set epoch
        if len(series):
            span = SegmentList([ts.span for ts in series]).extent()
            for ax in self.axes:
                ax.set_epoch(span[0])
                ax.set_xlim(*span)
                if not hasattr(self, '_auto_gps') or self._auto_gps:
                    ax.auto_gps_scale()
            for ax in self.axes[:-1]:
                ax.set_xlabel("")

    # -----------------------------------------------
    # properties

    @property
    def epoch(self):
        """Find the GPS epoch of this plot
        """
        axes = self._find_axes('timeseries')
        return axes.epoch

    def get_epoch(self):
        return self.epoch

    @auto_refresh
    def set_epoch(self, gps):
        """Set the GPS epoch of this plot
        """
        axeslist = self._find_all_axes('timeseries')
        for axes in axeslist:
            axes.set_epoch(gps)

    @property
    def gps_scale(self):
        axes = self._find_axes('timeseries')
        return axes.gps_scale

    def get_gps_scale(self):
        return self.gps_scale

    def set_gps_scale(self, scale):
        axeslist = self._find_all_axes('timeseries')
        for axes in axeslist:
            axes.set_gps_scale(scale)


    # -----------------------------------------------
    # extend add_timseries

    def add_timeseries(self, timeseries, **kwargs):
        super(TimeSeriesPlot, self).add_timeseries(timeseries, **kwargs)
        if not self.epoch:
            self.set_epoch(timeseries.epoch)
            

    @auto_refresh
    def set_time_format(self, format_='gps', epoch=None, scale=None,
                        autoscale=True, addlabel=True):
        """Set the time format for this plot.

        Currently, only the 'gps' format is accepted.

        Parameters
        ----------
        format_ : `str`
            name of the time format
        epoch : :class:`~astropy.time.core.Time`, optional
            GPS start epoch for the time axis
        scale : `float`, optional
            overall scaling for axis ticks in seconds, e.g. 60 shows
            minutes from the epoch
        autoscale : `bool`, optional
            auto-scale the axes when the format is set
        addlabel : `bool`, optional
            auto-set a default label for the x-axis

        Returns
        -------
        TimeFormatter
            the :class:`~gwpy.plotter.ticks.TimeFormatter` for this axis
        """
        if epoch and not scale:
            duration = self.xlim[1] - self.xlim[0]
            for scale in ticks.GPS_SCALE.keys()[::-1]:
               if duration > scale*4:
                   break
        formatter = ticks.TimeFormatter(format=format_, epoch=epoch,
                                        scale=scale)
        self.axes.xaxis.set_major_formatter(formatter)
        locator = ticks.AutoTimeLocator(epoch=epoch, scale=scale)
        self.axes.xaxis.set_major_locator(locator)
        self.axes.fmt_xdata = lambda t: LIGOTimeGPS(t)
        if addlabel:
            self.xlabel = ("Time (%s) from %s (%s)"
                           % (formatter.scale_str_long,
                              re.sub('\.0+', '', self.epoch.utc.iso),
                              self.epoch.gps))
        if autoscale:
            self.axes.autoscale_view()
        return formatter

    def refresh(self):
        super(TimeSeriesPlot, self).refresh()
        for ax in self._find_all_axes('timeseries'):
            if not hasattr(ax, '_auto_gps') or ax._auto_gps == True:
                ax.auto_gps_scale()


class TimeSeriesAxes(Axes):
    """Extension of the basic matplotlib :class:`~matplotlib.axes.Axes`
    specialising in time-series display
    """
    name = 'timeseries'
    def __init__(self, *args, **kwargs):
        """Instantiate a new TimeSeriesAxes suplot
        """
        epoch = kwargs.pop('epoch', 0)
        scale = kwargs.pop('scale', 1)
        super(TimeSeriesAxes, self).__init__(*args, **kwargs)
        self.set_epoch(epoch)
        # set x-axis format
        if not kwargs.has_key('sharex') or kwargs['sharex'] is None:
            formatter = ticks.TimeFormatter(format='gps', epoch=epoch,
                                            scale=scale)
            self.xaxis.set_major_formatter(formatter)
            locator = ticks.AutoTimeLocator(epoch=epoch, scale=scale)
            self.xaxis.set_major_locator(locator)
            self.fmt_xdata = lambda t: LIGOTimeGPS(t)
            self.add_epoch_label()
            self.autoscale_view()

    # -----------------------------------------------
    # properties

    @property
    def epoch(self):
        """Find the GPS epoch of this plot
        """
        return self._epoch

    @auto_refresh
    def set_epoch(self, gps):
        """Set the GPS epoch of this plot
        """
        # set new epoch
        if gps is None or isinstance(gps, Time):
            self._epoch = gps
        else:
            if isinstance(gps, datetime.datetime):
                from lal import gpstime
                self._epoch = float(gpstime.utc_to_gps(gps))
            elif isinstance(gps, basestring):
                from lal import gpstime
                self._epoch = float(gpstime.str_to_gps(gps))
            self._epoch = Time(float(gps), format='gps')
        # update x-axis ticks and labels
        formatter = self.xaxis.get_major_formatter()
        if isinstance(formatter, ticks.TimeFormatter):
            locator = self.xaxis.get_major_locator()
            oldepoch = formatter.epoch
            formatter.epoch = locator.epoch = self._epoch
            formatter.set_locs(locator.refresh())
            # update xlabel
            oldiso = re.sub('\.0+', '', oldepoch.utc.iso)
            xlabel = self.xlabel.get_text()
            if xlabel:
                if re.search(oldiso, xlabel):
                    self.xlabel = xlabel.replace(
                                         oldiso, re.sub('\.0+', '',
                                                        self.epoch.utc.iso))
                xlabel = self.xlabel.get_text()
                if re.search(str(oldepoch.gps), xlabel):
                    self.xlabel = xlabel.replace(str(oldepoch.gps),
                                                 str(self.epoch.gps))

    @auto_refresh
    def add_epoch_label(self):
        formatter = self.xaxis.get_major_formatter()
        if isinstance(formatter, ticks.TimeFormatter):
            scale = formatter.scale_str_long
        else:
            scale = 'seconds'
        utc = re.sub('\.0+', '', self.epoch.utc.iso)
        gps = self.epoch.gps
        return self.set_xlabel('Time (%s) from %s (%s)' % (scale, utc, gps))

    @property
    def gps_scale(self):
        try:
            return self.axaxis.get_major_formatter().scale
        except AttributeError:
            return 1

    @auto_refresh
    def set_gps_scale(self, scale):
        """Set the GPS scale of this plot
        """
        formatter = self.xaxis.get_major_formatter()
        locator = self.xaxis.get_major_locator()
        if not isinstance(formatter, ticks.TimeFormatter):
            raise TypeError("Formatter of type '%s' does not have a scale. "
                            "Please try using a TimeFormatter instead"
                            % formatter.__class__.__name__)
        s = formatter.scale_str_long
        formatter.scale = scale
        locator.scale = scale
        xlabel = self.xlabel.get_text()
        if xlabel:
            if re.search(s, xlabel):
                self.xlabel = xlabel.replace(s, formatter.scale_str_long)
        self._auto_gps = False

    @auto_refresh
    def auto_gps_scale(self):
        """Automagically set the GPS scale for the time-axis of this plot
        based on the current view limits
        """
        duration = self.viewLim.x1 - self.viewLim.x0
        for s in ticks.GPS_SCALE.keys()[::-1]:
            if duration >= (10 * s):
                self.set_gps_scale(s)
                self._auto_gps = True
                return

    # -------------------------------------------
    # GWpy class plotting methods

    @auto_refresh
    def plot(self, *args, **kwargs):
        """Plot data onto these Axes.

        Parameters
        ----------
        args
            a single :class:`~gwpy.timeseries.core.TimeSeries` (or sub-class)
            or standard (x, y) data arrays
        kwargs
            keyword arguments applicable to :meth:`~matplotib.axes.Axes.plot`

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer

        See Also
        --------
        :meth:`~matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        from ..timeseries import TimeSeries
        from ..spectrogram import Spectrogram
        if len(args) == 1 and isinstance(args[0], TimeSeries):
            return self.plot_timeseries(*args, **kwargs)
        elif len(args) == 1 and isinstance(args[0], Spectrogram):
            return self.plot_spectrogram(*args, **kwargs)
        else:
            return super(TimeSeriesAxes, self).plot(*args, **kwargs)

    @auto_refresh
    def plot_timeseries(self, timeseries, **kwargs):
        """Plot a :class:`~gwpy.timeseries.core.TimeSeries` onto these
        axes

        Parameters
        ----------
        timeseries : :class:`~gwpy.timeseries.core.TimeSeries`
            data to plot
        **kwargs
            any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.plot`

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer

        See Also
        --------
        :meth:`~matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        if tex.USE_TEX:
            kwargs.setdefault('label', tex.label_to_latex(timeseries.name))
        else:
            kwargs.setdefault('label', timeseries.name)
        if not self.epoch.gps:
            self.set_epoch(timeseries.epoch)
        line = self.plot(timeseries.times, timeseries.data, **kwargs)
        if len(self.lines) == 1:
            self.set_xlim(*timeseries.span)
            self.auto_gps_scale()
        return line

    @auto_refresh
    def plot_timeseries_mmm(self, mean_, min_=None, max_=None, **kwargs):
        """Plot a `TimeSeries` onto these axes, with (min, max) shaded
        regions

        The `mean_` `TimeSeries` is plotted normally, while the `min_`
        and `max_ `TimeSeries` are plotted lightly below and above,
        with a fill between them and the mean_.

        Parameters
        ----------
        mean_ : :class:`~gwpy.timeseries.core.TimeSeries`
            data to plot normally
        min_ : :class:`~gwpy.timeseries.core.TimeSeries`
            first data set to shade to mean_
        max_ : :class:`~gwpy.timeseries.core.TimeSeries`
            second data set to shade to mean_
        **kwargs
            any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.plot`

        Returns
        -------
        artists : `tuple`
            a 5-tuple containing (Line2d for mean_, `Line2D` for min_,
            `PolyCollection` for min_ shading, `Line2D` for max_, and
            `PolyCollection` for max_ shading)

        See Also
        --------
        :meth:`~matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        # plot mean
        line1 = self.plot_timeseries(mean_, **kwargs)[0]
        # plot min and max
        kwargs.pop('label', None)
        color = kwargs.pop('color', line1.get_color())
        linewidth = kwargs.pop('linewidth', line1.get_linewidth()) / 2
        if min_ is not None:
            a = self.plot(min_.times, min_.data, color=color,
                          linewidth=linewidth, **kwargs)
            b = self.fill_between(min_.times, mean_.data, min_.data, alpha=0.1,
                                  color=color)
        else:
            a = b = None
        if max_ is not None:
            c = self.plot(max_.times, max_.data, color=color,
                          linewidth=linewidth, **kwargs)
            d = self.fill_between(max_.times, mean_.data, max_.data, alpha=0.1,
                                  color=color)
        else:
            c = d = None
        return (line1, a, b, c, d)

    @auto_refresh
    def plot_spectrogram(self, spectrogram, **kwargs):
        """Plot a :class:`~gwpy.spectrogram.core.Spectrogram` onto
        these axes

        Parameters
        ----------
        spectrogram : :class:`~gwpy.spectrogram.core.Spectrogram`
            data to plot
        **kwargs
            any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.plot`

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer

        See Also
        --------
        :meth:`~matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        cmap = kwargs.pop('cmap', None)
        if cmap is None:
            cmap = copy.deepcopy(cm.jet)
            cmap.set_bad(cmap(0.0))
        kwargs['cmap'] = cmap
        norm = kwargs.pop('norm', None)
        if norm == 'log':
            vmin = kwargs.get('vmin', None)
            vmax = kwargs.get('vmax', None)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        kwargs['norm'] = norm
        if not self.epoch.gps:
            self.set_epoch(0)
            self.set_epoch(spectrogram.epoch)
        x = numpy.concatenate((spectrogram.times.data,
                               [spectrogram.span_x[-1].value]))
        y = numpy.concatenate((spectrogram.frequencies.data,
                               [spectrogram.y0.value +
                                spectrogram.dy.value * spectrogram.shape[1]]))
        X,Y = numpy.meshgrid(x, y)
        mesh = self.pcolormesh(X, Y, spectrogram.data.T, **kwargs)
        if len(self.collections) == 1:
            self.set_xlim(*map(numpy.float64, spectrogram.span_x))
            self.set_ylim(*map(numpy.float64, spectrogram.span_y))
        if not self.get_ylabel():
            self.add_label_unit(spectrogram.yunit, axis='y')
        return mesh

    @auto_refresh
    def plot_dqflag(self, flag, y=None, **kwargs):
        """Plot a :class:`~gwpy.segments.flag.DataQualityFlag`
        onto these axes

        Parameters
        ----------
        flag : :class:`~gwpy.segments.flag.DataQualityFlag`
            data-quality flag to display
        y : `float`, optional
            y-axis value for new segments
        height : `float`, optional, default: 0.8
            height for each segment block
        **kwargs
            any other keyword arguments acceptable for
            :class:`~matplotlib.patches.Rectangle`

        Returns
        -------
        collection : :class:`~matplotlib.patches.PatchCollection`
            list of :class:`~matplotlib.patches.Rectangle` patches
        """
        if y is None:
            y = len(self.collections)
        name = ':'.join([str(attr) for attr in
                         (flag.ifo, flag.name, flag.version) if
                         attr is not None])
        try:
            if not self.epoch.gps:
                self.set_epoch(flag.valid[0][0])
            else:
                self.set_epoch(min(self.epoch.gps, flag.valid[0][0]))
        except IndexError:
            pass
        return self.plot_segmentlist(flag.active, y=y, label=name, **kwargs)

    @auto_refresh
    def plot_segmentlist(self, segmentlist, y=None, **kwargs):
        """Plot a :class:`~gwpy.segments.segments.SegmentList` onto
        these axes

        Parameters
        ----------
        segmentlist : :class:`~gwpy.segments.segments.SegmentList`
            list of segments to display
        y : `float`, optional
            y-axis value for new segments
        **kwargs
            any other keyword arguments acceptable for
            :class:`~matplotlib.patches.Rectangle`

        Returns
        -------
        collection : :class:`~matplotlib.patches.PatchCollection`
            list of :class:`~matplotlib.patches.Rectangle` patches
        """
        if y is None:
            y = len(self.collections)
        patches = []
        for seg in segmentlist:
            patches.append(self.build_segment(seg, y, **kwargs))
        try:
            if not self.epoch.gps:
                self.set_epoch(segmentlist[0][0])
            else:
                self.set_epoch(min(self.epoch.gps, segmentlist[0][0]))
        except IndexError:
            pass
        return self.add_collection(PatchCollection(patches, True))

    @auto_refresh
    def plot_segmentlistdict(self, segmentlistdict, y=None, dy=1, **kwargs):
        """Plot a :class:`~gwpy.segments.segments.SegmentListDict` onto
        these axes

        Parameters
        ----------
        segmentlistdict : :class:`~gwpy.segments.segments.SegmentListDict`
            (name, :class:`~gwpy.segments.segments.SegmentList`) dict
        y : `float`, optional
            starting y-axis value for new segmentlists
        **kwargs
            any other keyword arguments acceptable for
            :class:`~matplotlib.patches.Rectangle`

        Returns
        -------
        collections : `list`
            list of :class:`~matplotlib.patches.PatchCollection` sets for
            each segmentlist
        """
        if y is None:
            y = len(self.collections)
        collections = []
        for name,segmentlist in segmentlistdict.iteritems():
            collections.append(self.plot_segmentlist(segmentlist, y=y,
                                                     label=name, **kwargs))
            y += dy
        return collections

    @staticmethod
    def build_segment(segment, y, height=.8, valign='center', **kwargs):
        """Build a :class:`~matplotlib.patches.Rectangle` to display
        a single :class:`~gwpy.segments.segments.Segment`

        Parameters
        ----------
        segment : :class:`~gwpy.segments.segments.Segment`
            [start, stop) GPS segment
        y : `float`
            y-axis peosition for segment
        height : `float`, optional, default: 1
            height (in y-axis units) for segment
        valign : `str`
            alignment of segment on y-axis value:
            `top`, `center`, or `bottom`
        **kwargs
            any other keyword arguments acceptable for
            :class:`~matplotlib.patches.Rectangle`

        Returns
        -------
        box : `~matplotlib.patches.Rectangle`
            rectangle patch for segment display
        """
        if valign.lower() == 'center':
            y0 = y - height/2.
        elif valign.lower() == 'top':
            y0 = y - height
        elif valign.lower() != 'bottom':
            raise ValueError("valign must be one of 'top', 'center', or "
                             "'bottom'")
        return Rectangle((segment[0], y), width=abs(segment), height=height,
                         **kwargs)


register_projection(TimeSeriesAxes)
