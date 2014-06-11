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

"""This module defines a locators/formatters and a scale for GPS data.
"""

import re

import numpy

from matplotlib import (ticker, docstring)
from matplotlib.scale import (register_scale, LinearScale,
                              get_scale_docs, get_scale_names)
from matplotlib.transforms import Transform

from astropy import units

from ..time import Time
from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

TIME_UNITS = [units.nanosecond,
              units.microsecond,
              units.millisecond,
              units.second,
              units.minute,
              units.hour,
              units.day,
              units.week,
              units.year,
              units.kiloyear,
              units.megayear,
              units.gigayear]


# ---------------------------------------------------------------------------
# Define re-usable scale mixin

class GPSMixin(object):
    """Mixin adding GPS-related attributes to any class.
    """
    def __init__(self, *args, **kwargs):
        self.set_unit(kwargs.pop('unit', None))
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
        elif isinstance(epoch, Time):
            self._epoch = epoch.utc.gps
        else:
            self._epoch = float(epoch)

    epoch = property(fget=get_epoch, fset=set_epoch,
                     doc=get_epoch.__doc__)

    def get_unit(self):
        """GPS step scale for this formatter
        """
        return self._unit

    def set_unit(self, unit):
        # default None to seconds
        if unit is None:
            unit = units.second
        # accept all core time units
        if isinstance(unit, units.NamedUnit) and unit.physical_type == 'time':
            self._unit = unit
            return
        # convert float to custom unit in seconds
        if isinstance(unit, float):
            unit = units.Unit(unit * units.second)
        # otherwise, should be able to convert to a time unit
        try:
            unit = units.Unit(unit)
        except ValueError as e:
            # catch annoying plurals
            try:
                unit = units.Unit(str(unit).rstrip('s'))
            except ValueError:
                raise e
        # decompose and check that it's actually a time unit
        u = unit.decompose()
        if u._bases != [units.second]:
            raise ValueError("Cannot set GPS unit to %s" % unit)
        # check equivalent units
        for other in TIME_UNITS:
            if other.decompose().scale == u.scale:
                self._unit = other
                return
        raise ValueError("Unrecognised unit: %s" % unit)

    unit = property(fget=get_unit, fset=set_unit,
                    doc=get_unit.__doc__)

    def get_unit_name(self):
        if not self.unit:
            return None
        name = sorted(self.unit.names, key=lambda u: len(u))[-1]
        if len(name) == 1:
            return name
        else:
            return '%ss' % name

    def get_scale(self):
        """The scale (in seconds) of the current GPS unit.
        """
        return self.unit.decompose().scale

    scale = property(fget=get_scale, doc=get_scale.__doc__)


# ---------------------------------------------------------------------------
# Define GPS transforms

class GPSTransformBase(GPSMixin, Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True
    is_affine = True
    has_inverse = True


class GPSTransform(GPSTransformBase):
    """Transform GPS time into N * scale from epoch.
    """
    def transform_non_affine(self, a):
        """Transform an array of GPS times.
        """
        if self.epoch is None:
             return numpy.round(a / self.scale, 4)
        else:
             return numpy.round((a - self.epoch) / self.scale, 4)

    def inverted(self):
        return InvertedGPSTransform(unit=self.unit, epoch=self.epoch)


class InvertedGPSTransform(GPSTransform):
    """Transform time (scaled units) from epoch into GPS time.
    """
    def transform_non_affine(self, a):
        if self.epoch is None:
            return numpy.round(a * self.scale, 4)
        else:
            return numpy.round(a * self.scale + self.epoch, 4)

    def inverted(self):
        return GPSTransform(unit=self.unit, epoch=self.epoch)

# ---------------------------------------------------------------------------
# Define GPS locators and formatters


class GPSLocatorMixin(GPSMixin):
    """Metaclass for GPS-axis locator
    """
    def tick_values(self, vmin, vmax):
        trans = self.axis._scale.get_transform()
        vmin = trans.transform(vmin)
        vmax = trans.transform(vmax)
        locs = super(GPSLocatorMixin, self).tick_values(vmin, vmax)
        return trans.inverted().transform(locs)

    def refresh(self):
        """refresh internal information based on current lim
        """
        return self()


class GPSAutoLocator(GPSLocatorMixin, ticker.MaxNLocator):
    """Find the best position for ticks on a given axis from the data.

    This auto-locator gives a simple extension to the matplotlib
    `~matplotlib.ticker.AutoLocator` allowing for variations in scale
    and zero-time epoch.
    """
    def __init__(self, epoch=None, unit=1, nbins=12, steps=None,
                 **kwargs):
        """Initialise a new `AutoTimeLocator`, optionally with an `epoch`
        and a `scale` (in seconds).

        Each of the `epoch` and `scale` keyword arguments should match those
        passed to the `~gwpy.plotter.GPSFormatter`
        """
        if not steps and unit == 3600:
            steps = [1, 2, 4, 5, 6, 8, 10, 12, 24]
        elif not steps:
            steps = [1, 2, 5, 10]
        super(GPSAutoLocator, self).__init__(epoch=epoch, unit=unit,
                                             nbins=nbins, steps=steps,
                                             **kwargs)


class GPSAutoMinorLocator(GPSLocatorMixin, ticker.AutoMinorLocator):
    def __call__(self):
        """Return the locations of the ticks
        """
        majorlocs = self.axis.get_majorticklocs()
        trans = self.axis._scale.get_transform()
        try:
            majorstep = majorlocs[1] - majorlocs[0]
        except IndexError:
            # Need at least two major ticks to find minor tick locations
            # TODO: Figure out a way to still be able to display minor
            # ticks without two major ticks visible. For now, just display
            # no ticks at all.
            majorstep = 0

        if self.ndivs is None:
            if majorstep == 0:
                # TODO: Need a better way to figure out ndivs
                ndivs = 1
            else:
                scale_ = trans.get_scale()
                gpsstep = majorstep / scale_
                x = int(round(10 ** (numpy.log10(gpsstep) % 1)))
                if x in [1, 5, 10]:
                    ndivs = 5
                else:
                    ndivs = 4
        else:
            ndivs = self.ndivs

        minorstep = majorstep / ndivs

        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        if len(majorlocs) > 0:
            t0 = majorlocs[0]
            tmin = numpy.ceil((vmin - t0) / minorstep) * minorstep
            tmax = numpy.floor((vmax - t0) / minorstep) * minorstep
            locs = numpy.arange(tmin, tmax, minorstep) + t0
            cond = numpy.abs((locs - t0) % majorstep) > minorstep / 10.0
            locs = locs.compress(cond)
        else:
            locs = []

        return self.raise_if_exceeds(numpy.array(locs))
    pass


class GPSFormatter(GPSMixin, ticker.Formatter):
    """Locator for astropy Time objects
    """
    def __call__(self, t, pos=None):
        trans = self.axis._scale.get_transform()
        if isinstance(t, Time):
            t = t.gps
        f = trans.transform(t)
        if numpy.isclose(f, int(f)):
            f = int(f)
        return re.sub('\.0+\Z', '', str(f))


# ---------------------------------------------------------------------------
# Define GPS scale

class GPSScale(GPSMixin, LinearScale):
    """A GPS scale, displaying time (scaled units) from an epoch.
    """
    name = 'gps'
    GPSTransform = GPSTransform
    InvertedGPSTransform = InvertedGPSTransform

    def __init__(self, axis, unit=None, epoch=None):
        """
        unit:
            either name (`str`) or scale (float in seconds)
        """
        viewlim = axis.get_view_interval()
        #try and copy data from the last scale
        if isinstance(epoch, Time):
            epoch = epoch.utc.gps
        elif isinstance(epoch, units.Quantity):
            epoch = epoch.value
        if epoch is None and type(axis._scale) is GPSScale:
            epoch = axis._scale.get_epoch()
        # otherwise get from current view
        if epoch is None:
            epoch = viewlim[0]
        if unit is None and type(axis._scale) is GPSScale:
            unit = axis._scale.get_unit()
        if unit is None:
            duration = viewlim[1] - (min(viewlim[0], epoch))
            unit = units.second
            for u in TIME_UNITS[::-1]:
                if duration >= u.decompose().scale * 4:
                    unit = u
                    break
        super(GPSScale, self).__init__(unit=unit, epoch=epoch)
        self._transform = self.GPSTransform(unit=self.unit, epoch=self.epoch)

    def get_transform(self):
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(GPSAutoLocator(unit=self.unit,
                                              epoch=self.epoch))
        axis.set_major_formatter(GPSFormatter(unit=self.unit,
                                              epoch=self.epoch))
        axis.set_minor_locator(GPSAutoMinorLocator(epoch=self.epoch))
        axis.set_minor_formatter(ticker.NullFormatter())

register_scale(GPSScale)


class AutoGPSScale(GPSScale):
    """Automagic GPS scaling based on visible data
    """
    name = 'auto-gps'

register_scale(AutoGPSScale)


# register all the astropy time units that have sensible long names
def gps_scale_factory(unit):
    class FixedGPSScale(GPSScale):
        name = str('%ss' % unit.long_names[0])

        def __init__(self, axis, epoch=None):
            super(FixedGPSScale, self).__init__(axis, epoch=epoch, unit=unit)
    return FixedGPSScale

for _unit in TIME_UNITS:
    register_scale(gps_scale_factory(_unit))

# update the docstring for matplotlib scale methods
docstring.interpd.update(
    scale=' | '.join([repr(x) for x in get_scale_names()]),
    scale_docs=get_scale_docs().rstrip())
