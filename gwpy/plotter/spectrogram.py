# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""An extension of the Plot class for handling Spectrograms
"""

import re
import numpy

from ..time import Time
from ..spectrogram import Spectrogram
from .timeseries import TimeSeriesPlot
from . import ticks
from .utils import log_transform
from .decorators import auto_refresh

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['SpectrogramPlot']


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

        # plot time series
        logy = None
        for spectrogram in spectrograms:
            self.add_spectrogram(spectrogram, **plotargs)
            if logy is not None and spectrogram.logscale != logy:
                raise ValueError("Plotting linear and logscale Spectrograms "
                                 "on the same plot is not supported")
            logy = spectrogram.logscale
        self.logy = logy
        if len(spectrograms):
            self.epoch = min(spec.epoch for spec in spectrograms)
            self.set_time_format('gps', epoch=self.epoch)

    @property
    def ylim(self):
        if self.logy:
            return _log_lim(self.axes, self.axes.get_ylim(), 'y')
        else:
            return self.axes.get_ylim()
    @ylim.setter
    @auto_refresh
    def ylim(self, ylim):
        if self._logy:
            self.axes.set_ylim(*_effective_log_lim(self.axes, ylim, 'y'))
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

    # -----------------------------------------------
    # extend add_spectrogram

    def add_spectrogram(self, spectrogram, **kwargs):
        super(SpectrogramPlot, self).add_spectrogram(spectrogram, **kwargs)
        if not self.epoch:
            self.epoch = spectrogram.epoch
            self.set_time_format('gps', self.epoch)


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

