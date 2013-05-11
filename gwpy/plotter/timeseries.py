# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Plot data from a TimeSeries
"""

from . import (tex, ticks)
from .core import BasicPlot
from .decorators import auto_refresh

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


class TimeSeriesPlot(BasicPlot):
    """Plot data from a LAL TimeSeries object
    """
    def __init__(self, *series, **kwargs):
        # extract plotting keyword arguments
        plotargs = dict()
        plotargs["linewidth"] = kwargs.pop("linewidth", 2)
        plotargs["color"] = kwargs.pop("color", "black")
        plotargs["linestyle"] = kwargs.pop("linestyle", "-")
        # set default figure arguments
        kwargs.setdefault("figsize", [12,6])

        # initialise figure
        super(TimeSeriesPlot, self).__init__(**kwargs)
        self._series = []

        # plot time series
        for timeseries in series:
            self._series.append(timeseries)
            self.add_line(timeseries.get_times(), timeseries.data,
                          label=tex.label_to_latex(timeseries.name))
        self.set_time_format("gps", epoch=self.epoch)
        self.xlabel = ("Time from epoch (%s, %s)"
                      % (self.epoch, self.epoch.iso))
        self._ax.autoscale_view()

    @property
    def epoch(self):
        return min(t.epoch for t in self._series)

    @auto_refresh
    def set_time_format(self, format_, epoch=None, **kwargs): 
        formatter = ticks.TimeFormatter(format=format_, epoch=epoch, **kwargs)
        self._ax.xaxis.set_major_formatter(formatter)


    def set_tick_rotation(self, rotation=0, minor=False):
        if minor:
            ticks = self._ax.xaxis.get_minor_ticks()
        else:
            ticks = self._ax.xaxis.get_major_ticks()
        align = (rotation == 0 and 'center' or
                 rotation > 180 and 'left' or
                 'right')
        kwargs = {"rotation": rotation, "horizontalalignment": align}
        for i, tick in enumerate(ticks):
            if tick.label1On:
                tick.label1.update(kwargs)
            if tick.label2On:
                tick.label2.update(kwargs)

