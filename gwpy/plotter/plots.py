# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module provides a set of standard plotting classes extending
the core plot class of this package.
"""

import numpy

from .core import BasicPlot
from .tex import label_to_latex
from . import ticks

__author__ = "Duncan M. Macleod <duncan.macleod@ligo.org>"
__version__ = version.id
__date__ = version.date

GPS_UNIT = {'gps':1, 'seconds':1, 'minutes':60, 'hours':3600, 'days':86400}


class TimeSeriesPlot(BasicPlot):
    """A plot showing data from a time-series.
    """
    def __init__(self, *series, **kwargs):
        """Setup a new TimeSeriesPlot
        """
        kwargs.setdefault("figsize", [12, 6])
        super(TimeSeriesPlot, self).__init__(**kwargs)
        self._epoch = None
        self._series = {}
        for timeseries in series:
            self.add_series(timeseries)
            if not self._epoch:
                self._epoch = timeseries.epoch
            else:
                self._epoch = min(self._epoch, timeseries.epoch)
        self.set_time_format("gps")

    def add_series(self, series, **kwargs):
        self._series[series.name] = series
        x = (numpy.arange(series.data.length) * series.deltaT +
             float(series.epoch)).astype(LIGOTimeGPS)
        self.add_line(series.get_times(), series.data,
                      label=label_to_latex(series.name))

    def set_time_format(self, format_):
        """Set the time-like format for this TimeSeriesPlot

        @param format_
            the time-like format of choice, accepts
            - '`gps`' - GPS time
            - '`date`' - Date/time
            - '`seconds`' - number of seconds from epoch
            - '`minutes`' - number of minutes from epoch
            - '`hours`' - number of hours from epoch
            - '`days`' - number of days from epoch
        """
        format_ = format_.lower()
        # set default
        if self._xformat is None:
            self._xformat = format_
            return

        # transform data
        transform = ticks.transform_factory(self._xformat, format_)
        self.transform_axis("x", transform)

        # set tick format
        if format_ == "date":
            locator = ticks.dates.AutoDateLocator()
            locator.set_axis(self._ax.xaxis)
            locator.autoscale()
            formatter = ticks.AutoDateFormatter(locator)
            self._ax.xaxis.set_major_locator(locator)
            self._ax.xaxis.set_major_formatter(formatter)
            #ticks.set_tick_rotation(self._ax.xaxis, rotation=335)
            if not self.xlabel and self._epoch:
                epoch = (self._epoch.gpsNanoSeconds and "%f" % self._epoch or 
                         "%d" % self._epoch)
                self.xlabel = ("Start date/time: %s (%s)"
                               % (gpstime.gps_to_str(self._epoch), epoch))
        elif format_ in GPS_UNIT.keys():
            formatter = ticks.GPSFormatter(self._epoch, GPS_UNIT[format_])
            self._ax.xaxis.set_major_formatter(formatter)
            if not self.xlabel and self._epoch:
                epoch = (self._epoch.gpsNanoSeconds and "%f" % self._epoch or 
                         "%d" % self._epoch)
                self.xlabel = ("Time (%s) since %s (%s)"
                               % (format_, gpstime.gps_to_str(self._epoch),
                                  epoch))
        else:
            raise NotImplementedError("Only the GPSFormatter has been "
                                      "implemented so far")
        self._xformat = format_

