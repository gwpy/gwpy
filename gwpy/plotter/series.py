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
