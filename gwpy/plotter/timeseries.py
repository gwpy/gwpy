# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""An extension of the Plot class for handling TimeSeries
"""

import re
import datetime

from matplotlib import axes
from matplotlib.projections import register_projection

from lal import LIGOTimeGPS

from .core import Plot
from ..segments import SegmentList
from ..time import Time
from ..timeseries import TimeSeries
from . import ticks
from .axes import Axes
from .decorators import auto_refresh

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['TimeSeriesPlot']


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
        formatter = ticks.TimeFormatter(format='gps', epoch=epoch, scale=scale)
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
        print gps
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
            if re.search(oldiso, xlabel):
                self.xlabel = xlabel.replace(
                                     oldiso, re.sub('\.0+', '',
                                                    self.epoch.utc.iso))
            xlabel = self.xlabel.get_text()
            if re.search(str(oldepoch.gps), xlabel):
                self.xlabel = xlabel.replace(str(oldepoch.gps),
                                             str(self.epoch.gps))

    # -------------------------------------------
    # Axes methods

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
            ts = args[0]
            args = (ts.times, ts.data)
            kwargs.setdefault('label', ts.name)
            if not self.epoch.gps:
                self.set_epoch(ts.epoch)
        return super(TimeSeriesAxes, self).plot(*args, **kwargs)


register_projection(TimeSeriesAxes)
