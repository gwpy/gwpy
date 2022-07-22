# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2021)
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

"""This module defines locators/formatters and a scale for GPS data.
"""

from decimal import Decimal
from numbers import Number

import numpy

from matplotlib import (ticker, docstring)
from matplotlib.scale import (register_scale, LinearScale, get_scale_names)
from matplotlib.transforms import Transform
try:
    from matplotlib.scale import _get_scale_docs as get_scale_docs
except ImportError:  # matplotlib < 3.1
    from matplotlib.scale import get_scale_docs

from astropy import units

from ..time import (to_gps, from_gps)

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

GPS_SCALES = {}


def register_gps_scale(scale_class):
    """Register a new GPS scale.

    ``scale_class`` must be a subclass of `GPSScale`.
    """
    register_scale(scale_class)
    GPS_SCALES[scale_class.name] = scale_class


def _truncate(f, n):
    """Truncates/pads a float `f` to `n` decimal places without rounding

    From https://stackoverflow.com/a/783927/1307974 (CC-BY-SA)
    """
    s = str(f)
    if "e" in s or "E" in s:
        return f"{f:.{n}f}"
    i, p, d = s.partition(".")
    return ".".join([i, (d+"0"*n)[:n]])


# -- base mixin for all GPS manipulations -------------------------------------

class GPSMixin(object):
    """Mixin adding GPS-related attributes to any class.
    """
    def __init__(self, *args, **kwargs):
        self.set_unit(kwargs.pop('unit', None))
        self.set_epoch(kwargs.pop('epoch', None))
        try:  # call super for __init__ if this is part of a larger MRO
            super().__init__(*args, **kwargs)
        except TypeError:  # otherwise return
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
        if isinstance(epoch, (Number, Decimal)):
            self._epoch = float(epoch)
        else:
            self._epoch = float(to_gps(epoch))

    epoch = property(fget=get_epoch, fset=set_epoch,
                     doc=get_epoch.__doc__)

    def get_unit(self):
        """GPS step scale
        """
        return self._unit

    def set_unit(self, unit):
        """Set the GPS step scale
        """
        # accept all core time units
        if unit is None or (
            isinstance(unit, units.NamedUnit)
            and unit.physical_type == 'time'
        ):
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
            raise ValueError(f"cannot set GPS unit to '{unit}'")
        # check equivalent units
        for other in TIME_UNITS:
            if other.decompose().scale == dec.scale:
                self._unit = other
                return
        raise ValueError(f"unrecognised unit '{unit}'")

    unit = property(fget=get_unit, fset=set_unit,
                    doc=get_unit.__doc__)

    def get_unit_name(self):
        """Returns the name of the unit for this GPS scale

        Note that this returns a simply-pluralised version of the name.
        """
        if not self.unit:
            return None
        name = sorted(self.unit.names, key=len)[-1]
        return name + "s"  # pluralise

    def get_scale(self):
        """The scale (in seconds) of the current GPS unit.
        """
        if self.unit is None:
            return 1
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
        return super().transform(values)

    def transform_non_affine(self, values):
        """Transform an array of GPS times.

        This method is designed to filter out transformations that will
        generate text elements that require exact precision, and use
        `Decimal` objects to do the transformation, and simple `float`
        otherwise.
        """
        scale = self.scale or 1
        epoch = self.epoch or 0

        values = numpy.asarray(values)

        # handle simple or data transformations with floats
        if self._parents or (  # part of composite transform (from draw())
                epoch == 0  # no large additions
                and scale == 1  # no multiplications
        ):
            return self._transform(values, float(epoch), float(scale))

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
        vdec = Decimal(_truncate(value, 12))
        edec = Decimal(_truncate(epoch, 12))
        sdec = Decimal(_truncate(scale, 12))
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


class GPSLocatorMixin(object):
    """Metaclass for GPS-axis locator
    """
    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        trans = self.axis.get_transform()
        vmin = trans.transform(vmin)
        vmax = trans.transform(vmax)
        return trans.inverted().transform(self.tick_values(vmin, vmax))

    def refresh(self):
        """refresh internal information based on current lim
        """
        return self()


class GPSAutoLocator(ticker.MaxNLocator):
    """Find the best position for ticks on a given axis from the data.

    This auto-locator gives a simple extension to the matplotlib
    `~matplotlib.ticker.AutoLocator` allowing for variations in scale
    and zero-time epoch.
    """
    def __init__(self, nbins=12, steps=None, **kwargs):
        """Initialise a new `AutoTimeLocator`, optionally with an `epoch`
        and a `scale` (in seconds).

        Each of the `epoch` and `scale` keyword arguments should match those
        passed to the `GPSFormatter`
        """
        super().__init__(nbins=nbins, steps=steps, **kwargs)

    def tick_values(self, vmin, vmax):
        transform = self.axis.get_transform()
        unit = transform.get_unit()
        steps = self._steps

        vmin, vmax = transform.transform((vmin, vmax))

        # if less than 6 weeks, major tick every week
        if steps is None and unit == units.week and vmax - vmin <= 6:
            self.set_params(steps=[1, 10])
        else:
            self.set_params(steps=None)

        try:
            ticks = super().tick_values(vmin, vmax)
        finally:
            self._steps = steps
        return transform.inverted().transform(ticks)


class GPSAutoMinorLocator(GPSLocatorMixin, ticker.AutoMinorLocator):
    """Find the best position for minor ticks on a given GPS-scaled axis.
    """
    def __call__(self):
        """Return the locations of the ticks
        """
        majorlocs = self.axis.get_majorticklocs()
        trans = self.axis.get_transform()
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

        if numpy.size(majorlocs):
            epoch = majorlocs[0]
            tmin = numpy.floor((vmin - epoch) / minorstep) * minorstep
            tmax = numpy.ceil((vmax - epoch) / minorstep) * minorstep
            locs = numpy.arange(tmin, tmax, minorstep) + epoch
            cond = numpy.abs((locs - epoch) % majorstep) > minorstep / 10.0
            locs = locs.compress(cond)
        else:
            locs = []

        return self.raise_if_exceeds(numpy.array(locs))


class GPSFormatter(ticker.Formatter):
    """Locator for astropy Time objects
    """
    def __call__(self, t, pos=None):
        # transform using float() to get nicer
        trans = self.axis.get_transform()
        flt = trans.transform(float(t))
        if flt.is_integer():
            return int(flt)
        return flt


# -- scales -------------------------------------------------------------------

class GPSScale(GPSMixin, LinearScale):
    """A GPS scale, displaying time (scaled units) from an epoch.
    """
    name = 'auto-gps'
    GPSTransform = GPSTransform
    InvertedGPSTransform = InvertedGPSTransform

    def __init__(self, axis, unit=None, epoch=None):
        """
        unit:
            either name (`str`) or scale (float in seconds)
        """
        super().__init__(unit=unit, epoch=epoch)
        self.axis = axis
        # set tight scaling on parent axes
        getattr(axis.axes, f"set_{axis.axis_name}margin")(0)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(GPSAutoLocator())
        axis.set_major_formatter(GPSFormatter())
        axis.set_minor_locator(GPSAutoMinorLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    @staticmethod
    def _lim(axis):
        # if autoscaling and datalim is set, use it
        dlim = tuple(axis.get_data_interval())
        if (
                getattr(axis.axes, f"get_autoscale{axis.axis_name}_on")()
                and not numpy.isinf(dlim).any()
        ):
            return dlim
        # otherwise use the view lim
        return tuple(axis.get_view_interval())

    def _auto_epoch(self, axis):
        # use the lower data/view limit as the epoch
        epoch = round(self._lim(axis)[0])

        # round epoch in successive units for large scales
        unit = self.get_unit()
        date = from_gps(epoch)
        fields = ('second', 'minute', 'hour', 'day')
        for i, u in enumerate(fields[1:]):
            if unit < units.Unit(u):
                break
            if u in ('day',):
                date = date.replace(**{fields[i]: 1})
            else:
                date = date.replace(**{fields[i]: 0})
        return int(to_gps(date))

    def _auto_unit(self, axis):
        vmin, vmax = self._lim(axis)
        duration = vmax - vmin
        for scale in TIME_UNITS[::-1]:
            base = scale.decompose().scale
            # for large durations, prefer smaller units
            if scale > units.second:
                base *= 4
            # for smaller durations, prefer larger units
            else:
                base *= 0.01
            if duration >= base:
                return scale
        return units.second

    def get_transform(self):
        # get current settings
        epoch = self.get_epoch()
        unit = self.get_unit()

        # dynamically set epoch and/or unit if None
        if unit is None:
            self.set_unit(self._auto_unit(self.axis))
        if epoch is None:
            self.set_epoch(self._auto_epoch(self.axis))

        # build transform on-the-fly
        try:
            return self.GPSTransform(unit=self.get_unit(),
                                     epoch=self.get_epoch())
        finally:  # reset to current settings
            self.set_epoch(epoch)
            self.set_unit(unit)


register_gps_scale(GPSScale)


# register all the astropy time units that have sensible long names
def _gps_scale_factory(unit):
    """Construct a GPSScale for this unit
    """
    class FixedGPSScale(GPSScale):
        """`GPSScale` for a specific GPS time unit
        """
        name = (unit.long_names or unit.names)[0] + "s"

        def __init__(self, axis, epoch=None):
            """
            """
            super().__init__(axis, epoch=epoch, unit=unit)
    return FixedGPSScale


for _unit in TIME_UNITS:
    if _unit is units.kiloyear:  # don't go past 'year' for GPSScale
        break
    register_gps_scale(_gps_scale_factory(_unit))

# update the docstring for matplotlib scale methods
docstring.interpd.update(
    scale=' | '.join([repr(x) for x in get_scale_names()]),
    scale_docs=get_scale_docs().rstrip(),
)
