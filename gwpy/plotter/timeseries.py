# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""An extension of the Plot class for handling TimeSeries
"""

import re

from lal import LIGOTimeGPS

from ..time import Time
from ..timeseries import TimeSeries
from . import (Plot, ticks)
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
            self.epoch = min(ts.epoch for ts in series)
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
    def set_time_format(self, format_, epoch=None, scale=1.0,
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

