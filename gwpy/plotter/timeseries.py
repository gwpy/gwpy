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

"""An extension of the Plot class for handling TimeSeries
"""

import re
import datetime
import numpy
import copy

from matplotlib import (pyplot, cm, colors)
from matplotlib.projections import register_projection
from matplotlib.artist import allow_rasterization
from matplotlib.cbook import iterable

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    from mpl_toolkits.axes_grid import make_axes_locatable


from ..time import LIGOTimeGPS
from . import (gps, tex)
from .core import Plot
from ..segments import SegmentList
from ..time import Time
from .axes import Axes
from .decorators import auto_refresh

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['TimeSeriesPlot', 'TimeSeriesAxes']


class TimeSeriesAxes(Axes):
    """Custom `Axes` for a :class:`~gwpy.plotter.TimeSeriesPlot`.
    """
    name = 'timeseries'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('xscale', 'auto-gps')
        super(TimeSeriesAxes, self).__init__(*args, **kwargs)
        self.fmt_xdata = lambda t: LIGOTimeGPS(t)
        self.set_xlabel('_auto')

    @allow_rasterization
    def draw(self, *args, **kwargs):
        # dynamically set scaling
        if self.get_xscale() == 'auto-gps':
            self.auto_gps_scale()
        # dynamically set x-axis label
        nolabel = self.get_xlabel() == '_auto'
        if nolabel:
            self.auto_gps_label()
        # auto-detect GPS scales
        super(TimeSeriesAxes, self).draw(*args, **kwargs)
        # reset label
        if nolabel:
            self.set_xlabel('_auto')

    draw.__doc__ = Axes.draw.__doc__

    # -----------------------------------------------
    # GPS scaling

    def set_xscale(self, scale, *args, **kwargs):
        super(TimeSeriesAxes, self).set_xscale(scale, *args, **kwargs)
        if scale != 'auto-gps' and self.get_xlabel() == '_auto':
            self.set_xlabel('')

    def auto_gps_label(self):
        scale = self.xaxis._scale
        epoch = scale.get_epoch()
        if epoch is None:
            self.set_xlabel('GPS Time')
        else:
            unit = scale.get_unit_name()
            utc = re.sub('\.0+', '',
                         Time(epoch, format='gps', scale='utc').iso)
            self.set_xlabel('Time [%s] from %s UTC (%s)' % (unit, utc, epoch))

    def auto_gps_scale(self):
        """Automagically set the GPS scale for the time-axis of this plot
        based on the current view limits
        """
        self.xaxis._set_scale('auto-gps', epoch=self.get_epoch())

    def set_epoch(self, epoch):
        try:
            xscale = self.get_xscale()
        except AttributeError:
            pass
        else:
            self.xaxis._set_scale(xscale, epoch=epoch)

    def get_epoch(self):
        return self.xaxis._scale.get_epoch()

    epoch = property(fget=get_epoch, fset=set_epoch, doc=get_epoch.__doc__)

    def set_xlim(self, left=None, right=None, emit=True, auto=False, **kw):
        if right is None and iterable(left):
            left, right = left
        left = float(left)
        right = float(right)
        if 'gps' in self.get_xscale() and self.epoch is None:
            self.set_epoch(left)
        super(TimeSeriesAxes, self).set_xlim(left=left, right=right, emit=emit,
                                             auto=auto, **kw)
    set_xlim.__doc__ = Axes.set_xlim.__doc__

    # -------------------------------------------
    # GWpy class plotting methods

    @auto_refresh
    def plot(self, *args, **kwargs):
        """Plot data onto these Axes.

        Parameters
        ----------
        args
            a single :class:`~gwpy.timeseries.TimeSeries` (or sub-class)
            or standard (x, y) data arrays
        kwargs
            keyword arguments applicable to :meth:`~matplotib.axens.Axes.plot`

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer

        See Also
        --------
        :meth:`matplotlib.axes.Axes.plot`
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
        """Plot a :class:`~gwpy.timeseries.TimeSeries` onto these
        axes

        Parameters
        ----------
        timeseries : :class:`~gwpy.timeseries.TimeSeries`
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
        :meth:`matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        if tex.USE_TEX:
            kwargs.setdefault('label', tex.label_to_latex(timeseries.name))
        else:
            kwargs.setdefault('label', timeseries.name)
        if not self.epoch:
            self.set_epoch(timeseries.x0)
        line = self.plot(timeseries.times.value, timeseries.value, **kwargs)
        if len(self.lines) == 1 and timeseries.size:
            self.set_xlim(*timeseries.span)
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
        mean_ : :class:`~gwpy.timeseries.TimeSeries`
            data to plot normally
        min_ : :class:`~gwpy.timeseries.TimeSeries`
            first data set to shade to mean_
        max_ : :class:`~gwpy.timeseries.TimeSeries`
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
        :meth:`matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        # plot mean
        line1 = self.plot_timeseries(mean_, **kwargs)[0]
        # plot min and max
        kwargs.pop('label', None)
        color = kwargs.pop('color', line1.get_color())
        linewidth = kwargs.pop('linewidth', line1.get_linewidth()) / 2
        if min_ is not None:
            a = self.plot(min_.times.value, min_.value, color=color,
                          linewidth=linewidth, **kwargs)
            b = self.fill_between(min_.times.value, mean_.value, min_.value,
                                  alpha=0.1, color=color,
                                  rasterized=kwargs.get('rasterized'))
        else:
            a = b = None
        if max_ is not None:
            c = self.plot(max_.times.value, max_.value, color=color,
                          linewidth=linewidth, **kwargs)
            d = self.fill_between(max_.times.value, mean_.value, max_.value,
                                  alpha=0.1, color=color,
                                  rasterized=kwargs.get('rasterized'))
        else:
            c = d = None
        return line1, a, b, c, d

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
        :meth:`matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        # rescue grid settings
        grid = (self.xaxis._gridOnMajor, self.xaxis._gridOnMinor,
                self.yaxis._gridOnMajor, self.yaxis._gridOnMinor)

        norm = kwargs.pop('norm', None)
        if norm == 'log':
            vmin = kwargs.get('vmin', None)
            vmax = kwargs.get('vmax', None)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        kwargs['norm'] = norm
        if not self.epoch:
            self.set_epoch(spectrogram.x0)
        x = numpy.concatenate((spectrogram.times.value,
                               [spectrogram.span[-1]]))
        y = numpy.concatenate((spectrogram.frequencies.value,
                               [spectrogram.band[-1]]))
        X, Y = numpy.meshgrid(x, y, copy=False, sparse=True)
        mesh = self.pcolormesh(X, Y, spectrogram.value.T, **kwargs)
        if len(self.collections) == 1:
            self.set_xlim(*spectrogram.span)
            self.set_ylim(*spectrogram.band)
        if not self.get_ylabel():
            self.add_label_unit(spectrogram.yunit, axis='y')

        # reset grid
        if grid[0]:
            self.xaxis.grid(True, 'major')
        if grid[1]:
            self.xaxis.grid(True, 'minor')
        if grid[2]:
            self.yaxis.grid(True, 'major')
        if grid[3]:
            self.yaxis.grid(True, 'minor')
        return mesh

register_projection(TimeSeriesAxes)


class TimeSeriesPlot(Plot):
    """`Figure` for displaying a :class:`~gwpy.timeseries.TimeSeries`.

    Parameters
    ----------
    *series : `TimeSeries`
        any number of :class:`~gwpy.timeseries.TimeSeries` to
        display on the plot
    **kwargs
        other keyword arguments as applicable for the
        :class:`~gwpy.plotter.Plot`
    """
    _DefaultAxesClass = TimeSeriesAxes

    def __init__(self, *series, **kwargs):
        """Initialise a new TimeSeriesPlot
        """
        kwargs.setdefault('projection', self._DefaultAxesClass.name)
        kwargs.setdefault('figsize', [12, 6])
        # extract custom keyword arguments
        sep = kwargs.pop('sep', False)
        epoch = kwargs.pop('epoch', None)
        sharex = kwargs.pop('sharex', False)
        sharey = kwargs.pop('sharey', False)
        # separate keyword arguments
        axargs, plotargs = self._parse_kwargs(kwargs)

        # generate figure
        super(TimeSeriesPlot, self).__init__(**kwargs)

        # plot data
        axesdata = self._get_axes_data(series, sep=sep)
        for data in axesdata:
            ax = self._add_new_axes(**axargs)
            for ts in data:
                ax.plot(ts, **plotargs)
            if 'sharex' not in axargs and sharex is True:
                axargs['sharex'] = ax
            if 'sharey' not in axargs and sharey is True:
                axargs['sharey'] = ax
        axargs.pop('sharex', None)
        axargs.pop('sharey', None)
        axargs.pop('projection', None)

        # set epoch
        if len(self.axes):
            flatdata = [ts for data in axesdata for ts in data]
            span = SegmentList([ts.span for ts in flatdata]).extent()
            for ax in self.axes:
                for key, val in axargs.iteritems():
                    getattr(ax, 'set_%s' % key)(val)
                if epoch is not None:
                    ax.set_epoch(epoch)
                else:
                    ax.set_epoch(span[0])
                if 'xlim' not in axargs:
                    ax.set_xlim(*span)
                if 'label' in plotargs:
                    ax.legend()
            for ax in self.axes[:-1]:
                ax.set_xlabel("")

    # -----------------------------------------------
    # properties

    @property
    def epoch(self):
        """Find the GPS epoch of this plot
        """
        try:  # look for this class (allow for subclasses)
            axes = self._find_axes(self._DefaultAxesClass.name)
        except IndexError:  # look for base timeseries
            for ax in self.axes:
                if isinstance(ax, TimeSeriesAxes):
                    axes = ax
        return axes.epoch

    def get_epoch(self):
        return self.epoch

    @auto_refresh
    def set_epoch(self, gps):
        """Set the GPS epoch of this plot
        """
        axeslist = self.get_axes(self._DefaultAxesClass.name)
        for axes in axeslist:
            axes.set_epoch(gps)

    # -----------------------------------------------
    # TimeSeriesPlot methods

    def add_timeseries(self, timeseries, **kwargs):
        super(TimeSeriesPlot, self).add_timeseries(timeseries, **kwargs)
        if self.epoch is None:
            self.set_epoch(timeseries.epoch)

    def add_state_segments(self, segments, ax=None, height=0.2, pad=0.1,
                           location='bottom', plotargs=dict()):
        """Add a `SegmentList` to this `TimeSeriesPlot` indicating state
        information about the main Axes data.

        By default, segments are displayed in a thin horizontal set of Axes
        sitting immediately below the x-axis of the main

        Parameters
        ----------
        segments : :class:`~gwpy.segments.flag.DataQualityFlag`
            A data-quality flag, or `SegmentList` denoting state segments
            about this Plot
        ax : `Axes`
            specific Axes set against which to anchor new segment Axes
        plotargs
            keyword arguments passed to
            :meth:`~gwpy.plotter.SegmentAxes.plot`
        """
        from .segments import SegmentAxes
        if not ax:
            try:
                ax = self.get_axes(self._DefaultAxesClass.name)[-1]
            except IndexError:
                raise ValueError("No 'timeseries' Axes found, cannot anchor "
                                 "new segment Axes.")
        pyplot.setp(ax.get_xticklabels(), visible=False)
        # add new axes
        if ax.get_axes_locator():
            divider = ax.get_axes_locator()._axes_divider
        else:
            divider = make_axes_locatable(ax)
        if location not in ['top', 'bottom']:
            raise ValueError("Segments can only be positoned at 'top' or "
                             "'bottom'.")
        segax = divider.append_axes(location, height, pad=pad,
                                    axes_class=SegmentAxes, sharex=ax)
        segax.set_xscale(ax.get_xscale())

        # plot segments and set axes properties
        segax.plot(segments, **plotargs)
        segax.grid(b=False, which='both', axis='y')
        segax.autoscale(axis='y', tight=True)
        # set ticks and label
        segax.set_xlabel(ax.get_xlabel())
        ax.set_xlabel("")
        segax.set_xlim(*ax.get_xlim())
        return segax
