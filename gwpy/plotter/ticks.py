# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines a number of tick locators for different data formats
"""

import re
from math import (ceil, floor, modf)
from numpy import arange

from matplotlib import (units as munits, ticker as mticker, pyplot, transforms as mtransforms)
from matplotlib.dates import (HOURS_PER_DAY, MINUTES_PER_DAY, SECONDS_PER_DAY,
                              SEC_PER_MIN, SEC_PER_HOUR, SEC_PER_DAY,
                              SEC_PER_WEEK, WEEKDAYS)

from astropy import time as atime

from ..time import Time
from .. import version

from astropy.utils import OrderedDict

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
        if scale and not epoch:
            raise ValueError("The GPS epoch must be stated if data scaling "
                             "is required")
        if scale is not None:
            self._scale = float(scale)
        else:
            self._scale = None

    def bin_boundaries(self, vmin, vmax):
        """Returns the boundaries for the ticks for this AutoTimeLocator
        """
        print self._scale, vmin, vmax
        if self._scale == 3600:
             scale = (vmax-vmin) >= 36 and 4 or (vmax-vmin) > 8 and 2 or 1
             low = floor(vmin)
             while low % scale:
                 low -= 1
             return arange(low, ceil(vmax)+1, scale)
        else:
             return super(AutoTimeLocator, self).bin_boundaries(vmin, vmax)

    def tick_values(self, vmin, vmax):
        """Return the ticks for this axis
        """
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=1e-13,
                                                         tiny=1e-14)
        locs = self.bin_boundaries(vmin, vmax)
        print locs
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
        if self.epoch:
            locs += float(self.epoch.gps)
        return self.raise_if_exceeds(locs)

    def get_view_interval(self):
        vmin, vmax = self.axis.get_view_interval()
        if self.epoch:
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



class TimeFormatter(mticker.Formatter):
    """Locator for astropy Time objects
    """
    def __init__(self, format='gps', epoch=None, scale=1.0):
        self._format = format
        self._tex = pyplot.rcParams["text.usetex"]
        if epoch and not isinstance(epoch, Time):
            self.epoch = Time(float(epoch), format=format)
        else:
            self.epoch = epoch
        self._scale = scale
        try:
            self.scale_str_long,self.scale_str_short = GPS_SCALE[scale]
        except KeyError:
            self.scale_str_long,self.scale_str_short = GPS_SCALE[1]

    def __call__(self, t, pos=None):
        if isinstance(t, Time):
            t = t.gps
        if self._format not in ['iso']:
            if self.epoch is not None:
                t = (t - self.epoch.gps)
            if self._scale is not None:
                t /= float(self._scale)
            t = round(float(t), 6)
        t = re.sub('.0+\Z', '', str(t))
        return t

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


def transform_factory(informat, outformat):
    """Transform data in a collection from one format to another
    """
    if informat == outformat:
        return lambda x: x
    if informat == "gps":
        if outformat == "date":
            return lambda x: dates.date2num(gpstime.gps_to_utc(LIGOTimeGPS(x)))
    raise NotImplementedError("Transform from GPS format to '%s' has "
                              "not been implemented" % outformat)


def transform(layer, axis, func):
    # transform data
    if isinstance(layer, lines.Line2D):
        line_transform(layer, axis, func)
    elif isinstance(layer, collections.Collection):
        path_transform(layer, axis, func)
    else:
        raise NotImplementedError("Transforming '%s' objects has not been "
                                  "implemented yet." % layer.__class__)


def line_transform(layer, axis, func):
    if axis == "x":
        layer.set_xdata(map(func, layer.get_xdata()))
    else:
        layer.set_xdata(map(func, layer.get_xdata()))


def path_transform(layer, axis, informat, outformat):
    data = layer.get_offsets()
    if axis == "x":
        data[:,0] = map(func, data[:,0])
    elif axis == "y":
        data[:,1] = map(func, data[:,1])
    else:
         raise ValueError("axis='%s' not understood, please give 'x' or 'y'"
                          % axis)
    layer.set_offsets(data)


def set_tick_rotation(axis, rotation=0, minor=False):
    if minor:
            ticks = axis.get_minor_ticks()
    else:
        ticks = axis.get_major_ticks()
    align = rotation > 180 and "left" or "right"
    kwargs = {"rotation": rotation, "horizontalalignment": align}
    for i, tick in enumerate(ticks):
        if tick.label1On:
            tick.label1.update(kwargs)
        if tick.label2On:
            tick.label2.update(kwargs)
