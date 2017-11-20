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

"""This module provides plotting utilities for visualising GW data

The standard data types (`TimeSeries`, `Table`, `DataQualityFlag`, ...) can
all be easily visualised using the relevant plotting objects, with
many configurable parameters both interactive, and in saving to disk.
"""

from matplotlib import pyplot

# utilities
from .rc import DEFAULT_PARAMS as GWPY_PLOT_PARAMS
from . import (  # pylint: disable=unused-import
    gps,  # GPS timing scales and formats
    log,  # Logarithimic scaling mods
)

# figure and axes extensions
from .core import Plot
from .axes import Axes
from .timeseries import (TimeSeriesPlot, TimeSeriesAxes)
from .spectrogram import (SpectrogramPlot)
from .frequencyseries import (FrequencySeriesPlot, FrequencySeriesAxes)
from .segments import (SegmentPlot, SegmentAxes)
from .filter import (BodePlot)
from .table import (EventTablePlot, EventTableAxes)
from .histogram import (HistogramPlot, HistogramAxes)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# pyplot.figure replacement
def figure(*args, **kwargs):  # pylint: disable=missing-docstring
    kwargs.setdefault('FigureClass', Plot)
    return pyplot.figure(*args, **kwargs)
figure.__doc__ = pyplot.figure.__doc__
