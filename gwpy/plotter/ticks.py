# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines a number of tick locators for different data formats
"""

import re
from math import modf

from matplotlib import (units as munits, ticker as mticker, pyplot, transforms as mtransforms)
from matplotlib.dates import (HOURS_PER_DAY, MINUTES_PER_DAY, SECONDS_PER_DAY,
                              SEC_PER_MIN, SEC_PER_HOUR, SEC_PER_DAY,
                              SEC_PER_WEEK, WEEKDAYS)

from astropy import time as atime

from ..time import Time
from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

TIME_FORMATS = atime.TIME_FORMATS.keys()


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
        if epoch and not isinstance(epoch, Time):
            self._epoch = Time(float(epoch), format='gps')
        else:
            self._epoch = epoch
        if scale is not None:
            self._scale = float(scale)
        else:
            self._scale = None

    def __call__(self):
        """Find the locations of ticks given the current view limits
        """
        vmin, vmax = self.axis.get_view_interval()
        if self._epoch:
            vmin -= self._epoch.gps
            vmax -= self._epoch.gps
        if self._scale:
            vmin /= self._scale
            vmax /= self._scale
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander = 0.05)
        locs = self.bin_boundaries(vmin, vmax)
        #print 'locs=', locs
        prune = self._prune
        if prune=='lower':
            locs = locs[1:]
        elif prune=='upper':
            locs = locs[:-1]
        elif prune=='both':
            locs = locs[1:-1]
        if self._scale:
            locs *= self._scale
        if self._epoch:
            locs += self._epoch.gps
        return self.raise_if_exceeds(locs)


class TimeFormatter(mticker.Formatter):
    """Locator for astropy Time objects
    """
    def __init__(self, format="gps", epoch=None, **kwargs):
        #super(TimeFormatter, self).__init__()
        #mticker.Formatter.__init__()
        self._format = format
        self._tex = pyplot.rcParams["text.usetex"]
        if epoch and not isinstance(epoch, Time):
            self._epoch = Time(float(epoch), format='gps')
        else:
            self._epoch = epoch
        self._scale = kwargs.pop('scale', 1.)
        self._t_args = kwargs

    def __call__(self, x, pos=None):
        t = Time(*modf(x)[::-1], format="gps",
                 **self._t_args).copy(self._format)
        if self._format not in ['iso']:
            if self._epoch is not None:
                t = (t - self._epoch).sec
            if self._scale is not None:
                t /= float(self._scale)
            t = round(float(t), 6)
        t = re.sub('.0+\Z', '', str(t))
        return t

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
