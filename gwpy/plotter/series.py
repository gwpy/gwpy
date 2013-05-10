# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines plotting classes for the data series defined in
`~gwpy.data`
"""

import warnings
from matplotlib import pyplot

from .utils import *
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
            self.add_timeseries(timeseries, **plotargs)
        if len(series):
            self.set_time_format("gps", epoch=self.epoch)
            self.xlabel = ("Time from epoch: %s (%s)"
                           % (self.epoch.iso.rstrip('0.'), self.epoch.gps))
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


class SpectrumPlot(BasicPlot):
    """Plot data from a LAL TimeSeries object
    """
    def __init__(self, *series, **kwargs):
        # extract plotting keyword arguments
        plotargs = dict()
        plotargs["linewidth"] = kwargs.pop("linewidth", 2)
        plotargs["color"] = kwargs.pop("color", "black")
        plotargs["linestyle"] = kwargs.pop("linestyle", "-")

        # initialise figure
        super(SpectrumPlot, self).__init__(**kwargs)
        self._series = []

        # plot time series
        for spectrum in series:
            self._series.append(spectrum)
            f = spectrum.get_frequencies()
            self.add_spectrum(spectrum)
        if len(series) == 1:
            self.add_label_unit(f.unit, axis="x")
            self.add_label_unit(series[0].unit, axis="y")
        if len(series):
            self.logx = self.logy = True
            self._ax.autoscale_view()


class SpectrogramPlot(TimeSeriesPlot):
    """Plot data from a `~gwpy.data.Spectrogram` object
    """
    def __init__(self, *spectrograms, **kwargs):
        self._logy = False
        # extract plotting keyword arguments
        plotargs = dict()
        plotargs["vmin"] = kwargs.pop("vmin", None)
        plotargs["vmax"] = kwargs.pop("vmax", None)
        plotargs["interpolation"] = kwargs.pop("interpolation", None)
        plotargs["aspect"] = kwargs.pop("aspect", None)
        plotargs["extent"] = kwargs.pop("extent", None)
        # set default figure arguments
        kwargs.setdefault("figsize", [12,6])

        # initialise figure
        super(SpectrogramPlot, self).__init__(**kwargs)
        self._series = []

        # plot time series
        logy = None
        for spectrogram in spectrograms:
            self._series.append(spectrogram)
            self.add_spectrogram(spectrogram, **plotargs)
            if logy is not None and spectrogram.logscale != logy:
                raise ValueError("Plotting linear and logscale Spectrograms "
                                 "on the same plot is not supported")
            logy = spectrogram.logscale
        self.logy = logy
        if len(spectrograms):
            self.set_time_format("gps", epoch=self.epoch)
            self.xlabel = ("Time from epoch: %s (%s)"
                           % (self.epoch.iso.rstrip('0.'), self.epoch.gps))
            self._ax.autoscale_view()

    @property
    def ylim(self):
        if self.logy:
            return _log_lim(self._ax, self._ax.get_ylim(), 'y')
        else:
            return self._ax.get_ylim()
    @ylim.setter
    @auto_refresh
    def ylim(self, ylim):
        if self._logy:
            self._ax.set_ylim(*_effective_log_lim(self._ax, ylim, 'y'))
            self.logy = True
        else:
            self._ax.set_ylim(*ylim)

    @property
    def logy(self):
        return self._logy
    @logy.setter
    @auto_refresh
    def logy(self, log):
        if log is self._logy:
             return
        if log:
            yticks, ylabels, yminorticks = log_transform(self.ylim)
            if len(yticks) < 2:
               yticks = yminorticks
               ylabels = map(tex.float_to_latex, yticks)
        else:
            yticks = numpy.linspace(*self._ax.get_ylim(),
                                    num=len(self._ax.get_yticks()))
            ylabels = map(tex.float_to_latex, yticks)
        self._ax.set_yticks(yticks)
        self._ax.set_yticklabels(ylabels)
        self._logy = bool(log)


def _effective_log_lim(ax, log_lim, axis='y'):
    axis = getattr(ax, '%saxis' % axis)
    lin_range = axis.get_data_interval()
    log_range = numpy.log10(lin_range)
    new_lim = numpy.log10(log_lim)
    eff_new_lim = (new_lim - log_range[0]) / float(log_range[1]-log_range[0])
    return eff_new_lim * (lin_range[1]-lin_range[0]) + lin_range[0]


def _log_lim(ax, log_lim, axis='y'):
    axis = getattr(ax, '%saxis' % axis)
    lin_range = axis.get_data_interval()
    log_range = numpy.log10(lin_range)
    lin_lim = (log_lim - lin_range[0]) / (lin_range[1]-lin_range[0])
    return 10 ** (lin_lim * (log_range[1] - log_range[0]) + log_range[0])
