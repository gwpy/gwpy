# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines plotting classes for the data series defined in
`~gwpy.data`
"""

import re
import warnings
from matplotlib import pyplot

from .utils import *
from . import (tex, ticks)
from .core import Plot
from .timeseries import TimeSeriesPlot
from .decorators import auto_refresh
from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


class SpectrumPlot(Plot):
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
            self.axes.autoscale_view()


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
            epoch_str = self.epoch.iso
            if re.search('.0+\Z', epoch_str):
                epoch_str = epoch_str.rsplit('.', 1)[0]
            self.xlabel = ("Time from epoch: %s (%s)"
                           % (epoch_str, self.epoch.gps))
            self.axes.autoscale_view()

    @property
    def ylim(self):
        if self.logy:
            return _log_lim(self.axes, self._ax.get_ylim(), 'y')
        else:
            return self.axes.get_ylim()
    @ylim.setter
    @auto_refresh
    def ylim(self, ylim):
        if self._logy:
            self.axes.set_ylim(*_effective_log_lim(self._ax, ylim, 'y'))
            self.logy = True
        else:
            self.axes.set_ylim(*ylim)

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
            yticks = numpy.linspace(*self.axes.get_ylim(),
                                    num=len(self.axes.get_yticks()))
            ylabels = map(tex.float_to_latex, yticks)
        self.axes.set_yticks(yticks)
        self.axes.set_yticklabels(ylabels)
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
