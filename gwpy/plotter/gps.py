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

from decimal import Decimal
from numbers import Number

import numpy

from matplotlib import (ticker, docstring)
from matplotlib.scale import (register_scale, LinearScale,
                              get_scale_docs, get_scale_names)
from matplotlib.transforms import Transform

from astropy import units

from ..time import Time

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

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


# -- base mixin for all GPS manipulations -------------------------------------

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
        """The GPS epoch
        """
        return self._epoch

    def set_epoch(self, epoch):
        """Set the GPS epoch
        """
        if epoch is None:
            self._epoch = None
            return
        if isinstance(epoch, Time):
            epoch = epoch.utc.gps
        self._epoch = float(epoch)

    epoch = property(fget=get_epoch, fset=set_epoch,
                     doc=get_epoch.__doc__)

    def get_unit(self):
        """GPS step scale
        """
        return self._unit

    def set_unit(self, unit):
        """Set the GPS step scale
        """
        # default None to seconds
        if unit is None:
            unit = units.second
        # accept all core time units
        if isinstance(unit, units.NamedUnit) and unit.physical_type == 'time':
            self._unit = unit
            return
        # convert float to custom unit in seconds
        if isinstance(unit, Number):
            unit = units.Unit(unit * units.second)
        # otherwise, should be able to convert to a time unit
        try:
            unit = units.Unit(unit)
        except ValueError as exc:
            # catch annoying plurals
            try:
                unit = units.Unit(str(unit).rstrip('s'))
            except ValueError:
                raise exc
        # decompose and check that it's actually a time unit
        dec = unit.decompose()
        if dec.bases != [units.second]:
            raise ValueError("Cannot set GPS unit to %s" % unit)
        # check equivalent units
        for other in TIME_UNITS:
            if other.decompose().scale == dec.scale:
                self._unit = other
                return
        raise ValueError("Unrecognised unit: %s" % unit)

    unit = property(fget=get_unit, fset=set_unit,
                    doc=get_unit.__doc__)

    def get_unit_name(self):
        """Returns the name of the unit for this GPS scale
        """
        if not self.unit:
            return None
        name = sorted(self.unit.names, key=len)[-1]
        if len(name) == 1:
            return name
        return '%ss' % name

    def get_scale(self):
        """The scale (in seconds) of the current GPS unit.
        """
        return self.unit.decompose().scale

    scale = property(fget=get_scale, doc=get_scale.__doc__)


# -- GPS transforms ---------------------------------------------------------

class GPSTransformBase(GPSMixin, Transform):
    """`Transform` to convert GPS times to time since epoch (and vice-verse)

    This class uses the `decimal.Decimal` object to protect against precision
    errors when converting to and from GPS times that may have 19 significant
    digits, which is more than `float` can handle.

    There is some logic to _only_ use the slow decimal transforms when
    absolutely necessary, normally when transforming tick positions.
    """
    input_dims = 1
    output_dims = 1
    is_separable = True
    is_affine = True
    has_inverse = True

    def transform(self, values):
        # format ticks using decimal for precision display
        if isinstance(values, (Number, Decimal)):
            return self._transform_decimal(values, self.epoch or 0, self.scale)
        return super(GPSTransformBase, self).transform(values)

    def transform_non_affine(self, values):
        """Transform an array of GPS times.

        This method is designed to filter out transformations that will
        generate text elements that require exact precision, and use
        `Decimal` objects to do the transformation, and simple `float`
        otherwise.
        """
        scale = self.scale
        epoch = self.epoch

        # handle simple or data transformations with floats
        if any([
                epoch is None,  # no large additions
                scale == 1,  # no multiplications
                self._parents,  # part of composite transform (from draw())
        ]):
            return self._transform(values, epoch, scale)

        # otherwise do things carefully (and slowly) with Decimals
        # -- ideally this only gets called for transforming tick positions
        flat = values.flatten()

        def _trans(x):
            return self._transform_decimal(x, epoch, scale)

        return numpy.asarray(list(map(_trans, flat))).reshape(values.shape)

    @staticmethod
    def _transform(value, epoch, scale):
        # this is declared by the actual transform subclass
        raise NotImplementedError

    @classmethod
    def _transform_decimal(cls, value, epoch, scale):
        """Transform to/from GPS using `decimal.Decimal` for precision
        """
        vdec = Decimal(repr(value))
        edec = Decimal(repr(epoch))
        sdec = Decimal(repr(scale))
        return type(value)(cls._transform(vdec, edec, sdec))


class GPSTransform(GPSTransformBase):
    """Transform GPS time into N * scale from epoch.
    """
    @staticmethod
    def _transform(value, epoch, scale):
        # convert GPS into scaled time from epoch
        return (value - epoch) / scale

    def inverted(self):
        return InvertedGPSTransform(unit=self.unit, epoch=self.epoch)


class InvertedGPSTransform(GPSTransform):
    """Transform time (scaled units) from epoch into GPS time.
    """
    @staticmethod
    def _transform(value, epoch, scale):
        # convert scaled time from epoch back into GPS
        return value * scale + epoch

    def inverted(self):
        return GPSTransform(unit=self.unit, epoch=self.epoch)


# -- locators and formatters --------------------------------------------------


class GPSLocatorMixin(GPSMixin):
    """Metaclass for GPS-axis locator
    """
    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        trans = self.axis._scale.get_transform()
        vmin = trans.transform(vmin)
        vmax = trans.transform(vmax)
        return trans.inverted().transform(self.tick_values(vmin, vmax))

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
        # steps for a week scale are dynamically set in tick_values()
        if not steps and unit == units.hour:
            steps = [1, 2, 4, 5, 6, 8, 10]
        elif not steps and unit == units.year:
            steps = [1, 2, 4, 6]
        elif not steps:
            steps = [1, 2, 5, 10]
        super(GPSAutoLocator, self).__init__(epoch=epoch, unit=unit,
                                             nbins=nbins, steps=steps,
                                             **kwargs)

    def tick_values(self, vmin, vmax):
        # if less than 6 weeks, major tick every week
        if self.get_unit() == units.week and vmax - vmin <= 6:
            self._steps = [1, 10]
        # otherwise fall-back to normal multiples
        else:
            self._steps = [1, 2, 5, 10]
        return super(GPSAutoLocator, self).tick_values(vmin, vmax)


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
                if trans.unit == units.week and gpsstep == 1:
                    ndivs = 7
                elif trans.unit == units.year and gpsstep <= 1:
                    ndivs = 6
                elif trans.unit != units.day and x in [1, 5, 10]:
                    ndivs = 5
                else:
                    ndivs = 4
        else:
            ndivs = self.ndivs

        minorstep = majorstep / ndivs

        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        if majorlocs.size:
            epoch = majorlocs[0]
            tmin = numpy.ceil((vmin - epoch) / minorstep) * minorstep
            tmax = numpy.floor((vmax - epoch) / minorstep) * minorstep
            locs = numpy.arange(tmin, tmax, minorstep) + epoch
            cond = numpy.abs((locs - epoch) % majorstep) > minorstep / 10.0
            locs = locs.compress(cond)
        else:
            locs = []

        return self.raise_if_exceeds(numpy.array(locs))
    pass


class GPSFormatter(GPSMixin, ticker.Formatter):
    """Locator for astropy Time objects
    """
    def __call__(self, t, pos=None):
        # transform using float() to get nicer
        trans = self.axis._scale.get_transform()
        f = trans.transform(float(t))
        if f.is_integer():
            return int(f)
        return f


# -- scales -------------------------------------------------------------------

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
        # try and copy data from the last scale
        if isinstance(epoch, Time):
            epoch = epoch.utc.gps
        elif isinstance(epoch, units.Quantity):
            epoch = epoch.value
        if epoch is None and isinstance(axis._scale, GPSScale):
            epoch = axis._scale.get_epoch()
        # otherwise get from current view
        if epoch is None:
            epoch = viewlim[0]
        if unit is None:
            duration = float(viewlim[1] - (min(viewlim[0], epoch)))
            unit = units.second
            for scale in TIME_UNITS[::-1]:
                if duration >= scale.decompose().scale * 4:
                    unit = scale
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
def _gps_scale_factory(unit):
    """Construct a GPSScale for this unit
    """
    class FixedGPSScale(GPSScale):
        """`GPSScale` for a specific GPS time unit
        """
        name = str('%ss' % unit.long_names[0])

        def __init__(self, axis, epoch=None):
            super(FixedGPSScale, self).__init__(axis, epoch=epoch, unit=unit)
    return FixedGPSScale


for _unit in TIME_UNITS:
    register_scale(_gps_scale_factory(_unit))

# update the docstring for matplotlib scale methods
docstring.interpd.update(
    scale=' | '.join([repr(x) for x in get_scale_names()]),
    scale_docs=get_scale_docs().rstrip())
