# Copyright (C) Duncan Macleod (2013)
# coding=utf-8
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
from math import modf

from matplotlib import (ticker as mticker, pyplot, transforms as mtransforms)
from matplotlib.dates import (SEC_PER_MIN, SEC_PER_HOUR, SEC_PER_DAY,
                              SEC_PER_WEEK)

from ..time import Time
from .. import version

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

GPS_SCALE = OrderedDict([(1, ('seconds', 's', 1)),
                         (SEC_PER_MIN, ('minutes', 'mins', 10)),
                         (SEC_PER_HOUR, ('hours', 'hrs', 4)),
                         (SEC_PER_DAY, ('days', 'd', 7)),
                         (SEC_PER_WEEK, ('weeks', 'w', 4))])


class GPSMixin(object):
    """Mixin adding GPS-related attributes to a `Locator`.
    """
    def __init__(self, *args, **kwargs):
        self.set_scale(kwargs.pop('scale', 1))
        self.set_epoch(kwargs.pop('epoch', None))
        # call super for __init__ if this is part of a larger MRO
        try:
            super(GPSMixin, self).__init__(*args, **kwargs)
        # otherwise return
        except TypeError:
            pass

    def get_epoch(self):
        """Starting GPS time epoch for this formatter
        """
        return self._epoch

    def set_epoch(self, epoch):
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

    epoch = property(fget=get_epoch, fset=set_epoch, doc=get_epoch.__doc__)

    def get_scale(self):
        """GPS step scale for this formatter
        """
        return self._scale

    def set_scale(self, scale):
        if scale not in GPS_SCALE:
            raise ValueError("Cannot set arbitrary GPS scales, please select "
                             "one of: %s" % GPS_SCALE.keys())
        self._scale = float(scale)

    scale = property(fget=get_scale, fset=set_scale, doc=get_scale.__doc__)


class GPSLocatorMixin(GPSMixin):
    """Metaclass for GPS-axis locator
    """
    def tick_values(self, vmin, vmax):
        if self.epoch is not None:
            vmin -= float(self.epoch.gps)
            vmax -= float(self.epoch.gps)
        if self.scale:
            vmin /= self._scale
            vmax /= self._scale
        locs = super(GPSLocatorMixin, self).tick_values(vmin, vmax)
        if self.scale:
            locs *= self.scale
        if self.epoch is not None:
            locs += float(self.epoch.gps)
        return locs

    def refresh(self):
        """refresh internal information based on current lim
        """
        return self()


class GPSMaxNLocator(GPSLocatorMixin, mticker.MaxNLocator):
    """Find the best position for ticks on a given axis from the data.

    This auto-locator gives a simple extension to the matplotlib
    `~matplotlib.ticker.AutoLocator` allowing for variations in scale
    and zero-time epoch.
    """
    def __init__(self, epoch=None, scale=None, nbins=12,
                 steps=list([1, 2, 4, 5, 6, 8, 10, 12, 16, 24]), **kwargs):
        """Initialise a new `AutoTimeLocator`, optionally with an `epoch`
        and a `scale` (in seconds).

        Each of the `epoch` and `scale` keyword arguments should match those
        passed to the `~gwpy.plotter.ticks.TimeFormatter`
        """
        super(GPSMaxNLocator, self).__init__(epoch=epoch, scale=scale,
                                             nbins=nbins, steps=steps, **kwargs)
        if self.scale and self.epoch is None:
            raise ValueError("The GPS epoch must be stated if data scaling "
                             "is required")


class GPSMultipleLocator(GPSLocatorMixin, mticker.MultipleLocator):
    pass


class GPSFormatter(GPSMixin, mticker.Formatter):
    """Locator for astropy Time objects
    """
    def __call__(self, t, pos=None):
        if isinstance(t, Time):
            t = t.gps
        if self.epoch is not None:
            t = (t - self.epoch.gps)
        if self.scale is not None:
            t /= float(self.scale)
        t = round(float(t), 6)
        t = re.sub('.0+\Z', '', str(t))
        return t
