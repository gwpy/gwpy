# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""An extension of the Plot class for handling TimeSeries
"""

import re
import datetime

from matplotlib import axes
from matplotlib.projections import register_projection

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from lal import LIGOTimeGPS

from .core import Plot
from ..segments import SegmentList
from ..time import Time
from ..timeseries import TimeSeries
from . import ticks
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
    def __init__(self, *series, **kwargs):
        """Initialise a new TimeSeriesPlot
        """
        # set figure size for x-axis as time
        kwargs.setdefault('figsize', [12,6])
        # generate figure
        super(TimeSeriesPlot, self).__init__(**kwargs)
        self.epoch = None
        # set epoch
        for ts in series:
            self.add_timeseries(ts)
        if len(series):
            span = SegmentList([ts.span for ts in series]).extent()
            self.epoch = span[0]
            self.xlim = span
            self.set_time_format('gps', epoch=self.epoch)

    # -----------------------------------------------
    # properties

    @property
    def epoch(self):
        """Find the GPS epoch of this plot
        """
        return self._epoch
    @epoch.setter
    @auto_refresh
    def epoch(self, gps):
        """Set the GPS epoch of this plot
        """
        # set new epoch
        if gps is None:
            self._epoch = gps
        else:
            if isinstance(gps, Time):
                self._epoch = gps
            else:
                self._epoch = Time(float(gps), format='gps')
        # update x-axis ticks and labels
        formatter = self.axes.xaxis.get_major_formatter()
        if isinstance(formatter, ticks.TimeFormatter):
            locator = self.axes.xaxis.get_major_locator()
            oldepoch = formatter.epoch
            formatter.epoch = locator.epoch = self._epoch
            formatter.set_locs(locator.refresh())
            # update xlabel
            oldiso = re.sub('\.0+', '', oldepoch.utc.iso)
            xlabel = self.xlabel.get_text()
            if re.search(oldiso, xlabel):
                self.xlabel = xlabel.replace(
                                     oldiso, re.sub('\.0+', '',
                                                    self.epoch.utc.iso))
            xlabel = self.xlabel.get_text()
            if re.search(str(oldepoch.gps), xlabel):
                self.xlabel = xlabel.replace(str(oldepoch.gps),
                                             str(self.epoch.gps))

    # -----------------------------------------------
    # extend add_timseries

    def add_timeseries(self, timeseries, **kwargs):
        super(TimeSeriesPlot, self).add_timeseries(timeseries, **kwargs)
        if not self.epoch:
            self.epoch = timeseries.epoch
            self.set_time_format('gps', self.epoch)

    # -----------------------------------------------
    # set time axis as GPS

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
            self.set_xlabel("Time (%s) from %s (%s)"
                            % (formatter.scale_str_long,
                               re.sub('\.0+', '', self.epoch.utc.iso),
                               self.epoch.gps))
            self.autoscale_view()

    # -----------------------------------------------
    # properties

    @property
    def epoch(self):
        """Find the GPS epoch of this plot
        """
        return self._epoch

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
            else:
                self.set_xlabel("Time (%s) from %s (%s)"
                                % (formatter.scale_str_long,
                                   re.sub('\.0+', '', self.epoch.utc.iso),
                                   self.epoch.gps))

    # -------------------------------------------
    # GWpy class plotting methods

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
        if len(args) == 1 and isinstance(args[0], TimeSeries):
            return self.plot_timeseries(*args, **kwargs)
        else:
            return super(TimeSeriesAxes, self).plot(*args, **kwargs)

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
        kwargs.setdefault('label', timeseries.name)
        if not self.epoch.gps:
            self.set_epoch(timeseries.epoch)
        line = self.plot(timeseries.times, timeseries.data, **kwargs)
        if len(self.lines) == 1:
            self.set_xlim(*timeseries.span)
        return line

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
