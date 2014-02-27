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

"""This module defines a number of tick locators for different data formats
"""

import re
from math import (ceil, floor, log, modf)
from numpy import arange

from matplotlib import (units as munits, ticker as mticker, pyplot, transforms as mtransforms)
from matplotlib.dates import (HOURS_PER_DAY, MINUTES_PER_DAY, SECONDS_PER_DAY,
                              SEC_PER_MIN, SEC_PER_HOUR, SEC_PER_DAY,
                              SEC_PER_WEEK, WEEKDAYS)

from astropy import time as atime

from ..time import Time
from .. import version

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

GPS_SCALE = OrderedDict([(1, ('seconds', 's')),
                         (60, ('minutes', 'mins')),
                         (3600, ('hours', 'hrs')),
                         (86400, ('days', 'd'))])


class TimeConverter(munits.ConversionInterface):
    """Define a converter between `~astropy.time.Time` objects and
    something locatable on an axis

    This converter uses the GPS time value of the given
    `~astropy.time.Time`.
    """
    @staticmethod
    def convert(value, unit, axis):
        return round(value.gps, 6)

    def axisinfo(unit, axis):
        if unit != "time":
            return None
        return munits.AxisInfo(majloc=AutoTimeLocator(),
                               majfmt=AutoTimeFormatter(), label='time')

    def default_units(x, axis):
        return "time"
# register the converter with matplotlib
munits.registry[Time] = TimeConverter()


class AutoTimeLocator(mticker.AutoLocator):
    """Find the best position for ticks on a given axis from the data.

    This auto-locator gives a simple extension to the matplotlib
    `~matplotlib.ticker.AutoLocator` allowing for variations in scale
    and zero-time epoch.
    """
    def __init__(self, epoch=None, scale=None):
        """Initialise a new `AutoTimeLocator`, optionally with an `epoch`
        and a `scale` (in seconds).

        Each of the `epoch` and `scale` keyword arguments should match those
        passed to the `~gwpy.plotter.ticks.TimeFormatter`
        """
        mticker.AutoLocator.__init__(self)
        #super(AutoTimeLocator, self).__init__()
        self.epoch = epoch
        if scale and epoch is None:
            raise ValueError("The GPS epoch must be stated if data scaling "
                             "is required")
        if scale is not None:
            self._scale = float(scale)
        else:
            self._scale = None

    def bin_boundaries(self, vmin, vmax):
        """Returns the boundaries for the ticks for this AutoTimeLocator
        """
        if self._scale:
             N = (vmax - vmin)
             scale =  2 ** ceil(log(ceil(N / 12.), 2))
             low = floor(vmin)
             while low % scale:
                 low -= 1
             return arange(low, ceil(vmax)+1, scale)
        else:
             return mticker.AutoLocator.bin_boundaries(self, vmin, vmax)

    def tick_values(self, vmin, vmax):
        """Return the ticks for this axis
        """
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=1e-13,
                                                         tiny=1e-14)
        locs = self.bin_boundaries(vmin, vmax)
        prune = self._prune
        if prune == 'lower':
            locs = locs[1:]
        elif prune == 'upper':
            locs = locs[:-1]
        elif prune == 'both':
            locs = locs[1:-1]
        return self.raise_if_exceeds(locs)

    def __call__(self):
        """Find the locations of ticks given the current view limits
        """
        vmin, vmax = self.get_view_interval()
        locs = self.tick_values(vmin, vmax)
        if self._scale:
            locs *= self._scale
        if self.epoch is not None:
            locs += float(self.epoch.gps)
        return self.raise_if_exceeds(locs)

    def get_view_interval(self):
        vmin, vmax = self.axis.get_view_interval()
        if self.epoch is not None:
            vmin -= float(self.epoch.gps)
            vmax -= float(self.epoch.gps)
        if self._scale:
            vmin /= self._scale
            vmax /= self._scale
        return mtransforms.nonsingular(vmin, vmax, expander = 0.05)

    def refresh(self):
        """refresh internal information based on current lim
        """
        return self()

    @property
    def epoch(self):
        """Starting GPS time epoch for this formatter
        """
        return self._epoch
    @epoch.setter
    def epoch(self, epoch):
        if epoch is None:
           self._epoch = None
           return
        elif not isinstance(epoch, Time):
            if hasattr(epoch, "seconds"):
                epoch = [epoch.seconds, epoch.nanoseconds*1e-9]
            elif hasattr(epoch, "gpsSeconds"):
                epoch = [epoch.gpsSeconds, epoch.gpsNanoSeconds*1e-9]
            else:
                epoch = modf(epoch)[::-1]
            epoch = Time(*epoch, format='gps', precision=6)
        self._epoch = epoch.copy(format='gps')

    @property
    def scale(self):
        """GPS step scale for this formatter
        """
        return self._scale

    @scale.setter
    def scale(self, scale, short=None, long=None):
        self._scale = scale


class TimeFormatter(mticker.Formatter):
    """Locator for astropy Time objects
    """
    def __init__(self, format='gps', epoch=None, scale=1.0):
        self._format = format
        self._tex = pyplot.rcParams["text.usetex"]
        self.set_epoch(epoch)
        self.set_scale(scale)

    def __call__(self, t, pos=None):
        if isinstance(t, Time):
            t = t.gps
        if self._format not in ['iso']:
            if self.epoch is not None:
                t = (t - self.epoch.gps)
            if self.scale is not None:
                t /= float(self.scale)
            t = round(float(t), 6)
        t = re.sub('.0+\Z', '', str(t))
        return t

    def set_scale(self, scale, short=None, long=None):
        self.scale = scale
        try:
            self.scale_str_long,self.scale_str_short = GPS_SCALE[scale]
        except KeyError:
            self.scale_str_long,self.scale_str_short = GPS_SCALE[1]
        if short:
            self.scale_str_short = short
        if long:
            self.scale_str_long = long

    def set_epoch(self, epoch):
        if epoch is None:
            self.epoch = None
            return
        elif not isinstance(epoch, Time):
            if hasattr(epoch, "seconds"):
                epoch = [epoch.seconds, epoch.nanoseconds*1e-9]
            elif hasattr(epoch, "gpsSeconds"):
                epoch = [epoch.gpsSeconds, epoch.gpsNanoSeconds*1e-9]
            else:
                epoch = modf(epoch)[::-1]
            epoch = Time(*epoch, format='gps', precision=6)
        self.epoch = epoch.copy(format='gps')
