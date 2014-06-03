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

"""Utilties for plotting directly from frames
"""

from .. import version
from ..timeseries import TimeSeries
from .timeseries import TimeSeriesPlot

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ["GWFramePlot"]


class GWFramePlot(TimeSeriesPlot):
    """Utility to plot directly from gravitational wave frames.
    """
    def __init__(self, gwf, channel, **kwargs):
        # read the frame
        self._gwf = gwf
        self._channel = channel
        start = kwargs.pop("start", -1)
        end = kwargs.pop("end", -1)
        self.read_frame(self._gwf, channel, start=start, end=end)
        super(GWFramePlot, self).__init__(self._series, **kwargs)

    def read_frame(self, framefile, channel, start=None, end=None):
        """Read the data for the given channel from the frame file
        """
        if end and start:
            duration = end-start
        else:
            duration = None
        self._series = TimeSeries.read(framefile, channel, epoch=start,
                                       duration=duration)
