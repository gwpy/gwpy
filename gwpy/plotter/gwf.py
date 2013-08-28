# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Utilties for plotting directly from frames
"""

from .. import version
from ..data import TimeSeries
from .series import TimeSeriesPlot

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
