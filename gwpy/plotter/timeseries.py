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
import numpy

from matplotlib import (pyplot, colors, __version__ as mpl_version)
from matplotlib.projections import register_projection
from matplotlib.artist import allow_rasterization
from matplotlib.cbook import iterable

from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..time import (Time, LIGOTimeGPS, to_gps)
from ..segments import SegmentList
from . import text
from .core import Plot
from .axes import Axes
from .decorators import auto_refresh

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['TimeSeriesPlot', 'TimeSeriesAxes']

if mpl_version > '2.0':  # imshow respects log scaling in >=2.0
    USE_IMSHOW = True
else:
    USE_IMSHOW = False


class TimeSeriesAxes(Axes):
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

        # dynamically set x-axis label
        nolabel = self.get_xlabel() == '_auto'
        if nolabel:
            self.auto_gps_label()

        # draw
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
    set_xscale.__doc__ = Axes.set_xscale.__doc__

    def auto_gps_label(self):
        """Automatically set the x-axis label based on the current GPS scale
        """
        scale = self.xaxis._scale
        epoch = scale.get_epoch()
        if int(epoch) == epoch:
            epoch = int(epoch)
        if epoch is None:
            self.set_xlabel('GPS Time')
        else:
            unit = scale.get_unit_name()
            utc = re.sub(r'\.0+', '',
                         Time(epoch, format='gps', scale='utc').iso)
            self.set_xlabel('Time [%s] from %s UTC (%s)'
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
            self.set_xscale(xscale, epoch=to_gps(epoch))

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
            a single `~gwpy.timeseries.TimeSeries` (or sub-class)
            or standard (x, y) data arrays

        kwargs
            keyword arguments applicable to :meth:`~matplotib.axens.Axes.plot`

        Returns
        -------
        Line2D
            the `~matplotlib.lines.Line2D` for this line layer

        See Also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        from ..timeseries import TimeSeriesBase
        from ..spectrogram import Spectrogram
        if len(args) == 1 and isinstance(args[0], TimeSeriesBase):
            return self.plot_timeseries(*args, **kwargs)
        elif len(args) == 1 and isinstance(args[0], Spectrogram):
            return self.plot_spectrogram(*args, **kwargs)
        return super(TimeSeriesAxes, self).plot(*args, **kwargs)

    @auto_refresh
    def plot_timeseries(self, timeseries, **kwargs):
        """Plot a `~gwpy.timeseries.TimeSeries` onto these
        axes

        Parameters
        ----------
        timeseries : `~gwpy.timeseries.TimeSeries`
            data to plot

        **kwargs
            any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.plot`

        Returns
        -------
        line : `~matplotlib.lines.Line2D`
            the newly added line

        See Also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        kwargs.setdefault('label', text.to_string(timeseries.name))
        if not self.epoch:
            self.set_epoch(timeseries.x0)
        line = self.plot(timeseries.times.value, timeseries.value, **kwargs)
        if len(self.lines) == 1 and timeseries.size:
            self.set_xlim(*timeseries.xspan)
        if not self.get_ylabel():
            self.set_ylabel(text.unit_as_label(timeseries.unit))
        return line

    @auto_refresh
    def plot_timeseries_mmm(self, mean_, min_=None, max_=None, **kwargs):
        """Plot a `TimeSeries` onto these axes, with shaded regions

        The ``mean_`` `TimeSeries` is plotted normally, while the ``min_``
        and ``max_`` `TimeSeries` are plotted lightly below and above,
        with a fill between them and the ``mean_``.

        Parameters
        ----------
        mean_ : `~gwpy.timeseries.TimeSeries`
            data to plot normally

        min_ : `~gwpy.timeseries.TimeSeries`
            first data set to shade to ``mean_``

        max_ : `~gwpy.timeseries.TimeSeries`
            second data set to shade to ``mean_``

        **kwargs
            any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.plot`

        Returns
        -------
        artists : `tuple`
            a 5-tuple containing:

            - `~matplotlib.lines.Line2d` for ``mean_``,
            - `~matplotlib.lines.Line2D` for ``min_``,
            - `~matplotlib.collections.PolyCollection` for ``min_`` shading,
            - `~matplotlib.lines.Line2D` for ``max_``, and
            - `~matplitlib.collections.PolyCollection` for ``max_`` shading

        See Also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        # plot mean
        meanline = self.plot_timeseries(mean_, **kwargs)[0]
        # plot min and max
        kwargs.pop('label', None)
        color = kwargs.pop('color', meanline.get_color())
        linewidth = kwargs.pop('linewidth', meanline.get_linewidth()) / 2
        if min_ is not None:
            minline = self.plot(min_.times.value, min_.value, color=color,
                                linewidth=linewidth, **kwargs)
            mincol = self.fill_between(min_.times.value, mean_.value,
                                       min_.value, alpha=0.1, color=color,
                                       rasterized=kwargs.get('rasterized'))
        else:
            minline = mincol = None
        if max_ is not None:
            maxline = self.plot(max_.times.value, max_.value, color=color,
                                linewidth=linewidth, **kwargs)
            maxcol = self.fill_between(max_.times.value, mean_.value,
                                       max_.value, alpha=0.1, color=color,
                                       rasterized=kwargs.get('rasterized'))
        else:
            maxline = maxcol = None
        return meanline, minline, mincol, maxline, maxcol

    @auto_refresh
    def plot_spectrogram(self, spectrogram, imshow=USE_IMSHOW, **kwargs):
        """Plot a `~gwpy.spectrogram.core.Spectrogram` onto
        these axes

        Parameters
        ----------
        spectrogram : `~gwpy.spectrogram.core.Spectrogram`
            data to plot

        imshow : `bool`, optional
            if `True`, use :meth:`~matplotlib.axes.Axes.imshow` to render
            the spectrogram as an image, otherwise use
            :meth:`~matplotlib.axes.Axes.pcolormesh`, default is `True`
            with `matplotlib >= 2.0`, otherwise `False`.

        **kwargs
            any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.imshow` (if ``imshow=True``),
            or :meth:`~matplotlib.Axes.pcolormesh` (``imshow=False``)

        Returns
        -------
        layer : `~matplotlib.collections.QuadMesh`, `~matplotlib.images.Image`
            the layer for this spectrogram

        See Also
        --------
        matplotlib.axes.Axes.imshow
        matplotlib.axes.Axes.pcolormesh
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        # rescue grid settings
        grid = (self.xaxis._gridOnMajor, self.xaxis._gridOnMinor,
                self.yaxis._gridOnMajor, self.yaxis._gridOnMinor)

        # set normalisation
        norm = kwargs.pop('norm', None)
        if norm == 'log':
            vmin = kwargs.get('vmin', None)
            vmax = kwargs.get('vmax', None)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        kwargs['norm'] = norm

        # set epoch if not set
        if not self.epoch:
            self.set_epoch(spectrogram.x0)

        # plot with imshow
        if imshow:
            extent = tuple(spectrogram.xspan) + tuple(spectrogram.yspan)
            if extent[2] == 0.:  # hack out zero on frequency axis
                extent = extent[:2] + (1e-300,) + extent[3:]
            kwargs.setdefault('extent', extent)
            kwargs.setdefault('origin', 'lower')
            kwargs.setdefault('interpolation', 'none')
            kwargs.setdefault('aspect', 'auto')
            layer = self.imshow(spectrogram.value.T, **kwargs)
        # plot with pcolormesh
        else:
            x = numpy.concatenate((spectrogram.times.value,
                                   [spectrogram.span[-1]]))
            y = numpy.concatenate((spectrogram.frequencies.value,
                                   [spectrogram.band[-1]]))
            xcoord, ycoord = numpy.meshgrid(x, y, copy=False, sparse=True)
            layer = self.pcolormesh(xcoord, ycoord, spectrogram.value.T,
                                    **kwargs)

        # format axes
        if len(self.collections) == 1:
            self.set_xlim(*spectrogram.span)
            self.set_ylim(*spectrogram.band)
        if not self.get_ylabel():
            self.set_ylabel(text.unit_as_label(spectrogram.yunit))

        # reset grid
        if grid[0]:
            self.xaxis.grid(True, 'major')
        if grid[1]:
            self.xaxis.grid(True, 'minor')
        if grid[2]:
            self.yaxis.grid(True, 'major')
        if grid[3]:
            self.yaxis.grid(True, 'minor')
        return layer


register_projection(TimeSeriesAxes)


class TimeSeriesPlot(Plot):
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
            for series in data:
                ax.plot(series, **plotargs)
            if 'sharex' not in axargs and sharex is True:
                axargs['sharex'] = ax
            if 'sharey' not in axargs and sharey is True:
                axargs['sharey'] = ax
        axargs.pop('sharex', None)
        axargs.pop('sharey', None)
        axargs.pop('projection', None)

        # set epoch
        if self.axes:
            flatdata = [ts for data in axesdata for ts in data]
            span = SegmentList([ts.span for ts in flatdata]).extent()
            for ax in self.axes:
                for key, val in axargs.items():
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
