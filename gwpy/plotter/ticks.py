#!/usr/bin/env python

# Copyright (C) 2012 Duncan M. Macleod
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""This module defines a number of tick locators for different data formats
"""

import re 
from math import modf

from matplotlib import (units as munits, ticker as mticker, pyplot, transforms as mtransforms)
from matplotlib.dates import (HOURS_PER_DAY, MINUTES_PER_DAY, SECONDS_PER_DAY,
                              SEC_PER_MIN, SEC_PER_HOUR, SEC_PER_DAY,
                              SEC_PER_WEEK, WEEKDAYS)

from ..time import Time
from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


def gps_time(gps):
    """Convert GPS float to Time object
    """
    return Time(gps, format="gps")


class TimeConverter(munits.ConversionInterface):
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
munits.registry[Time] = TimeConverter()


class AutoTimeLocator(mticker.AutoLocator):
    def __init__(self, epoch=None):
        mticker.AutoLocator.__init__(self)
        #super(AutoTimeLocator, self).__init__()
        if epoch and not isinstance(epoch, Time):
            self._epoch = Time(float(epoch), format='gps')
        else:
            self._epoch = epoch

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        if self._epoch:
            vmin -= self._epoch.gps
            vmax -= self._epoch.gps
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
        if self._epoch:
            locs += self._epoch.gps
        return self.raise_if_exceeds(locs)


class TimeLocator(mticker.Locator):

    def __init__(self, *args, **kwargs):
        super(TimeLocator, self).__unit__(*args, **kwargs)

    def get_locator(self, tmin, tmax):
        duration = tmax - tmin
        if duration < 1000:
            unit = 1
        elif duration < 20000:
            unit = 60
        elif duration < 604800:
            unit = 3600
        elif duration <= 31 * 86400:
            unit = 86400
        elif duration < 20 * 7 * 86400:
            unit = 86400 * 7


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
        self._t_args = kwargs

    def __call__(self, x, pos=None):
        t = Time(*modf(x)[::-1], format="gps",
                 **self._t_args).copy(self._format)
        if self._format not in ['iso']:
            if self._epoch is not None:
                t = (t - self._epoch).sec
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
