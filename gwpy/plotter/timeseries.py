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
import warnings

from matplotlib import pyplot
from matplotlib.projections import register_projection
from matplotlib.artist import allow_rasterization
from matplotlib.cbook import iterable

from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..time import (Time, LIGOTimeGPS, to_gps)
from .series import (SeriesPlot, SeriesAxes)
from .decorators import auto_refresh

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['TimeSeriesPlot', 'TimeSeriesAxes']


class TimeSeriesAxes(SeriesAxes):
    """Custom `Axes` for a `~gwpy.plotter.TimeSeriesPlot`.
    """
    name = 'timeseries'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('xscale', 'auto-gps')
        super(TimeSeriesAxes, self).__init__(*args, **kwargs)
        self.fmt_xdata = LIGOTimeGPS
        self.set_xlabel('_auto')

    @allow_rasterization
    def draw(self, *args, **kwargs):
        # dynamically set scaling
        if self.get_xscale() == 'auto-gps':
            self.auto_gps_scale()

        # dynamically set epoch
        try:
            noepoch = self.get_epoch() is None
        except AttributeError:
            noepoch = False
        else:
            if noepoch:
                self.set_epoch(round(self.viewLim.x0))

        # dynamically set x-axis label
        nolabel = self.get_xlabel() == '_auto'
        if nolabel:
            self.auto_gps_label()

        # draw
        try:
            super(TimeSeriesAxes, self).draw(*args, **kwargs)
        finally:
            # reset
            if noepoch:
                self.set_epoch(None)
            if nolabel:
                self.set_xlabel('_auto')

    draw.__doc__ = SeriesAxes.draw.__doc__

    # -- GPS scaling --------------------------------

    def set_xscale(self, scale, *args, **kwargs):
        super(TimeSeriesAxes, self).set_xscale(scale, *args, **kwargs)
        if scale != 'auto-gps' and self.get_xlabel() == '_auto':
            self.set_xlabel('')
    set_xscale.__doc__ = SeriesAxes.set_xscale.__doc__

    def auto_gps_label(self):
        """Automatically set the x-axis label based on the current GPS scale
        """
        scale = self.xaxis._scale
        epoch = scale.get_epoch()
        if epoch is None:
            return self.set_xlabel('GPS Time')
        if int(epoch) == epoch:
            epoch = int(epoch)
        unit = scale.get_unit_name()
        utc = re.sub(r'\.0+', '', Time(epoch, format='gps', scale='utc').iso)
        return self.set_xlabel('Time [%s] from %s UTC (%s)'
                               % (unit, utc, repr(epoch)))

    def auto_gps_scale(self):
        """Automagically set the GPS scale for the time-axis of this plot
        based on the current view limits
        """
        self.set_xscale('auto-gps', epoch=self.get_epoch())

    def set_epoch(self, epoch):
        """Set the GPS epoch (t=0) for these axes
        """
        try:
            xscale = self.get_xscale()
        except AttributeError:
            pass
        else:
            epoch = to_gps(epoch) if epoch is not None else None
            self.set_xscale(xscale, epoch=epoch)

    def get_epoch(self):
        """Return the current GPS epoch (t=0)
        """
        return self.xaxis._scale.get_epoch()

    epoch = property(fget=get_epoch, fset=set_epoch, doc=get_epoch.__doc__)

    def set_xlim(self, left=None, right=None, emit=True, auto=False, **kw):
        if right is None and iterable(left):
            left, right = left
        left = float(to_gps(left))
        right = float(to_gps(right))
        super(TimeSeriesAxes, self).set_xlim(left=left, right=right, emit=emit,
                                             auto=auto, **kw)
    set_xlim.__doc__ = SeriesAxes.set_xlim.__doc__

    # -- methods --------------------------------

    @auto_refresh
    def plot(self, *args, **kwargs):
        """Plot data onto these Axes

        Parameters
        ----------
        args
            a single `~gwpy.timeseries.TimeSeries` (or sub-class),
            `~gwpy.spectrogram.Spectrogram` (or sub-class),
            or standard (x, y) data arrays

        kwargs
            keyword arguments applicable to :meth:`~matplotib.axes.Axes.plot`

        Returns
        -------
        Line2D
            the `~matplotlib.lines.Line2D` for this line layer

        See Also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        from ..spectrogram import Spectrogram

        # plot spectrogram
        if len(args) == 1 and isinstance(args[0], Spectrogram):
            return self.plot_spectrogram(*args, **kwargs)

        # SeriesAxes.plot takes care of everything else
        return super(TimeSeriesAxes, self).plot(*args, **kwargs)

    plot_timeseries = SeriesAxes.plot_series

    @auto_refresh
    def plot_timeseries_mmm(self, mean_, min_=None, max_=None, **kwargs):
        warnings.warn('plot_timeseries_mmm has been deprecated, please '
                      'use instead plot_mmm()', DeprecationWarning)
        return self.plot_mmm(mean_, min_=min_, max_=max_, **kwargs)

    plot_spectrogram = SeriesAxes.plot_array2d


register_projection(TimeSeriesAxes)


class TimeSeriesPlot(SeriesPlot):
    """`Figure` for displaying a `~gwpy.timeseries.TimeSeries`.

    Parameters
    ----------
    *series : `TimeSeries`
        any number of `~gwpy.timeseries.TimeSeries` to
        display on the plot

    **kwargs
        other keyword arguments as applicable for the
        `~gwpy.plotter.Plot`
    """
    _DefaultAxesClass = TimeSeriesAxes

    def __init__(self, *series, **kwargs):
        kwargs.setdefault('figsize', [12, 6])
        super(TimeSeriesPlot, self).__init__(*series, **kwargs)

    # -- properties ---------------------------------

    @property
    def epoch(self):
        """The GPS epoch of this plot
        """
        try:  # look for this class (allow for subclasses)
            axes = self._find_axes(self._DefaultAxesClass.name)
        except IndexError:  # look for base timeseries
            for ax in self.axes:
                if isinstance(ax, TimeSeriesAxes):
                    axes = ax
        return axes.epoch

    def get_epoch(self):
        """Return the GPS epoch of this plot
        """
        return self.epoch

    @auto_refresh
    def set_epoch(self, gps):
        """Set the GPS epoch of this plot
        """
        axeslist = self.get_axes(self._DefaultAxesClass.name)
        for axes in axeslist:
            axes.set_epoch(gps)

    # -- methods ------------------------------------

    def add_state_segments(self, segments, ax=None, height=0.2, pad=0.1,
                           location='bottom', plotargs=dict()):
        """Add a `SegmentList` to this `TimeSeriesPlot` indicating state
        information about the main Axes data.

        By default, segments are displayed in a thin horizontal set of Axes
        sitting immediately below the x-axis of the main

        Parameters
        ----------
        segments : `~gwpy.segments.flag.DataQualityFlag`
            A data-quality flag, or `SegmentList` denoting state segments
            about this Plot

        ax : `Axes`
            specific Axes set against which to anchor new segment Axes

        plotargs
            keyword arguments passed to
            :meth:`~gwpy.plotter.SegmentAxes.plot`
        """
        from .segments import SegmentAxes

        # get axes to anchor against
        if not ax:
            try:
                ax = self.get_axes(self._DefaultAxesClass.name)[-1]
            except IndexError:
                raise ValueError("No 'timeseries' Axes found, cannot anchor "
                                 "new segment Axes.")

        # add new axes
        if ax.get_axes_locator():
            divider = ax.get_axes_locator()._axes_divider
        else:
            divider = make_axes_locatable(ax)
        if location not in ['top', 'bottom']:
            raise ValueError("Segments can only be positoned at 'top' or "
                             "'bottom'.")
        segax = divider.append_axes(location, height, pad=pad,
                                    axes_class=SegmentAxes, sharex=ax,
                                    epoch=ax.get_epoch(), xlim=ax.get_xlim(),
                                    xlabel=ax.get_xlabel())

        # plot segments
        segax.plot(segments, **plotargs)
        segax.grid(b=False, which='both', axis='y')
        segax.autoscale(axis='y', tight=True)

        # update anchor axes
        pyplot.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")

        return segax
