# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""GPS axis locators, formatters, and scales."""

from __future__ import annotations

import contextlib
from decimal import Decimal
from numbers import Number
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy
from astropy import units
from matplotlib import ticker
from matplotlib.scale import (
    LinearScale,
    _get_scale_docs as get_scale_docs,
    get_scale_names,
    register_scale,
)
from matplotlib.transforms import Transform

try:
    from matplotlib import _docstring
except ImportError:  # maybe matplotlib >= 3.9?
    _docstring = None  # type: ignore[assignment]

from ..time import (
    from_gps,
    to_gps,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import (
        Any,
        Literal,
    )

    from astropy.units import UnitBase
    from matplotlib.axis import Axis
    from numpy.typing import ArrayLike

    from ..time import SupportsToGps
    from ..typing import UnitLike

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

#: Maximum number of week ticks to display
WEEK_SCALE_MAJOR_TICKS = 6

#: Supported time scales
TIME_UNITS = (
    units.nanosecond,
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
    units.gigayear,
)

GPS_SCALES = {}


def _truncate(f: float, n: int) -> str:
    """Truncates/pads a float `f` to `n` decimal places without rounding.

    From https://stackoverflow.com/a/783927/1307974 (CC-BY-SA)
    """
    s = str(f)
    if "e" in s or "E" in s:
        return f"{f:.{n}f}"
    i, p, d = s.partition(".")
    return ".".join([i, (d+"0"*n)[:n]])


# -- base mixin for all GPS manipulations

class GPSMixin:
    """Mixin adding GPS-related attributes to any class."""

    def __init__(
        self,
        *args: Any,  # noqa: ANN401
        unit: UnitBase | None = None,
        epoch: Number | Decimal | SupportsToGps | None = None,
        **kwargs,
    ) -> None:
        """Initialise a new GPS-scaled object."""
        self.set_unit(unit)
        self.set_epoch(epoch)

        # call super for __init__ if this is part of a larger MRO
        with contextlib.suppress(TypeError):
            super().__init__(*args, **kwargs)

    def get_epoch(self) -> float | None:
        """Return the GPS epoch."""
        return self._epoch

    def set_epoch(self, epoch: Number | Decimal | SupportsToGps | None) -> None:
        """Set the GPS epoch."""
        if epoch is None:
            self._epoch = None
        else:
            self._epoch = float(to_gps(epoch))

    epoch = property(
        fget=get_epoch,
        fset=set_epoch,
        doc=get_epoch.__doc__,
    )

    def get_unit(self) -> UnitBase | None:
        """GPS step scale."""
        return self._unit

    def set_unit(self, unit: UnitLike | Number | None) -> None:
        """Set the GPS step scale."""
        # accept all core time units
        if (
            unit is None
            or (isinstance(unit, units.NamedUnit) and unit.physical_type == "time")
        ):
            self._unit = unit
            return

        # convert float to custom unit in seconds
        if isinstance(unit, Number):
            unit = units.Unit(unit * units.second)

        # otherwise, should be able to convert to a time unit
        try:
            unit = units.Unit(unit)
        except ValueError:
            # catch annoying plurals
            unit = units.Unit(str(unit).rstrip("s"))
        unit = cast("UnitBase", unit)

        # decompose and check that it's actually a time unit
        dec = unit.decompose()
        if dec.bases != [units.second]:
            msg = f"cannot set GPS unit to '{unit}'"
            raise ValueError(msg)

        # check equivalent units
        for other in TIME_UNITS:
            if other.decompose().scale == dec.scale:
                self._unit = other
                return

        msg = f"unrecognised unit '{unit}'"
        raise ValueError(msg)

    unit = property(
        fget=get_unit,
        fset=set_unit,
        doc=get_unit.__doc__,
    )

    def get_unit_name(self) -> str | None:
        """Return the name of the unit for this GPS scale.

        Note that this returns a simply-pluralised version of the name.
        """
        if not self.unit:
            return None
        try:
            name = self.unit.long_names[0]
        except IndexError:
            name = self.unit.name
        if len(name) > 1:
            return name + "s"  # pluralise for humans
        return name

    def get_scale(self) -> float:
        """Return the scale (in seconds) of the current GPS unit."""
        if self.unit is None:
            return 1
        return self.unit.decompose().scale

    scale = property(fget=get_scale, doc=get_scale.__doc__)


# -- GPS transforms ----------------

class _GPSTransformBase(GPSMixin, Transform):
    """Transform GPS time into N * scale from epoch.

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

    def transform(self, values: ArrayLike) -> numpy.ndarray:
        """Transform an array of GPS times."""
        # format ticks using decimal for precision display
        if isinstance(values, Number | Decimal):
            return self._transform_decimal(values, self.epoch or 0, self.scale)
        return super().transform(values)

    def transform_non_affine(self, values: ArrayLike) -> numpy.ndarray:
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
            # no large additions
            epoch == 0
            # no multiplications
            and scale == 1
        ):
            return self._transform(values, float(epoch), float(scale))

        # otherwise do things carefully (and slowly) with Decimals
        # -- ideally this only gets called for transforming tick positions
        flat = values.flatten()

        def _trans(x: float) -> float:
            return self._transform_decimal(x, epoch, scale)

        return numpy.fromiter(
            map(_trans, flat),
            dtype=float,
            count=flat.size,
        ).reshape(values.shape)

    @staticmethod
    def _transform(
        value: numpy.ndarray,
        epoch: float,
        scale: float,
    ) -> numpy.ndarray:
        """Transform the GPS ``value`` into a scaled time relative to an epoch."""
        # convert GPS into scaled time from epoch
        return (value - epoch) / scale

    @classmethod
    def _transform_decimal(
        cls,
        value: float,
        epoch: float,
        scale: float,
    ) -> float:
        """Transform to/from GPS using `decimal.Decimal` for precision."""
        vdec = Decimal(_truncate(value, 12))
        edec = Decimal(_truncate(epoch, 12))
        sdec = Decimal(_truncate(scale, 12))
        return type(value)(cls._transform(vdec, edec, sdec))  # type: ignore[arg-type]


class GPSTransform(_GPSTransformBase):
    """Transform GPS into time (scaled units) from epoch."""

    def inverted(self) -> InvertedGPSTransform:
        """Return the inverse of this `GPSTransform`."""
        return InvertedGPSTransform(unit=self.unit, epoch=self.epoch)


class InvertedGPSTransform(_GPSTransformBase):
    """Transform time (scaled units) from epoch into GPS time."""

    @staticmethod
    def _transform(
        value: numpy.ndarray,
        epoch: float,
        scale: float,
    ) -> numpy.ndarray:
        """Transform the scaled time back into GPS."""
        return value * scale + epoch

    def inverted(self) -> GPSTransform:
        """Return the inverse of this `InvertedGPSTransform`."""
        return GPSTransform(unit=self.unit, epoch=self.epoch)


# -- locators and formatters ---------

class GPSAutoLocator(ticker.MaxNLocator):
    """Find the best position for ticks on a given axis from the data.

    This auto-locator gives a simple extension to the matplotlib
    `~matplotlib.ticker.AutoLocator` allowing for variations in scale
    and zero-time epoch.
    """

    def __init__(
        self,
        nbins: int | Literal["auto"] | None = 12,
        **kwargs,
    ) -> None:
        """Initialise a new `GPSAutoLocator`.

        Each of the `epoch` and `scale` keyword arguments should match those
        passed to the `GPSFormatter`
        """
        super().__init__(
            nbins=nbins,
            **kwargs,
        )

    def tick_values(self, vmin: float, vmax: float) -> Sequence[float]:
        """Generate the list of tick values for the given interval."""
        self.axis: Axis
        transform = self.axis.get_transform()
        unit = transform.get_unit()
        steps = self._steps

        vmin, vmax = transform.transform((vmin, vmax))

        # if less than 6 weeks, major tick every week
        if (
            steps is None
            and unit == units.week
            and vmax - vmin <= WEEK_SCALE_MAJOR_TICKS
        ):
            self.set_params(steps=[1, 10])
        else:
            self.set_params(steps=None)

        try:
            ticks = super().tick_values(vmin, vmax)
        finally:
            self._steps = steps
        return transform.inverted().transform(ticks)


class GPSAutoMinorLocator(ticker.AutoMinorLocator):
    """Find the best position for minor ticks on a given GPS-scaled axis."""

    def __call__(self) -> Sequence[float]:
        """Return the locations of the ticks."""
        self.axis: Axis
        majorlocs = self.axis.get_majorticklocs()
        trans = self.axis.get_transform()
        try:
            majorstep = majorlocs[1] - majorlocs[0]
        except IndexError:
            # Need at least two major ticks to find minor tick locations
            # TODO (@duncanmmacleod): Figure out a way to still be able to
            # display minor ticks without two major ticks visible.
            # For now, just display no ticks at all.
            majorstep = 0

        if self.ndivs is None:
            if majorstep == 0:
                # TODO (@duncanmmacleod): Need a better way to figure out ndivs
                ndivs = 1
            else:
                scale_ = trans.get_scale()
                gpsstep = majorstep / scale_
                x = round(10 ** (numpy.log10(gpsstep) % 1))
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
    """Format ticks on a `GPSScale` axis."""

    def __call__(
        self,
        t: float,
        pos: int | None = None,  # noqa: ARG002
    ) -> str:
        """Format a tick on this scale."""
        trans = self.axis.get_transform()
        flt = float(trans.transform(float(t)))
        if flt.is_integer():
            return str(int(flt))
        return str(flt)


# -- scales --------------------------

class GPSScale(GPSMixin, LinearScale):
    """A GPS scale, displaying time (scaled units) from an epoch.

    Parameters
    ----------
    axis : `matplotlib.axis.Axis`.
        The axis to scale.

    unit : `astropy.units.Unit`, optional
        The unit to use for ticks on the axis.

    epoch : `float`, `gwpy.time.LIGOTimeGPS`, optional
        The GPS epoch (origin) for axis ticks.
    """

    name = "auto-gps"
    Transform = GPSTransform
    InvertedTransform = InvertedGPSTransform

    def __init__(
        self,
        axis: Axis,
        unit: UnitBase | None = None,
        epoch: Number | Decimal | SupportsToGps | None = None,
    ) -> None:
        """Initialise this `GPSScale`."""
        super().__init__(unit=unit, epoch=epoch)
        self.axis = axis
        # set tight scaling on parent axes
        getattr(axis.axes, f"set_{axis.axis_name}margin")(0)

    def set_default_locators_and_formatters(self, axis: Axis) -> None:
        """Set the defualt locators and formatters for ``axis``."""
        axis.set_major_locator(GPSAutoLocator())
        axis.set_major_formatter(GPSFormatter())
        axis.set_minor_locator(GPSAutoMinorLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    @staticmethod
    def _lim(axis: Axis) -> tuple[float, float]:
        """Find the current view limits of this ``axis``."""
        # if autoscaling and datalim is set, use it
        dlim = axis.get_data_interval()
        if (
            getattr(axis.axes, f"get_autoscale{axis.axis_name}_on")()
            and not numpy.isinf(dlim).any()
        ):
            return dlim

        # otherwise use the view lim
        return axis.get_view_interval()

    def _auto_epoch(self, axis: Axis) -> int:
        """Find the best GPS epoch (origin) for this ``axis``."""
        # use the lower data/view limit as the epoch
        epoch = round(self._lim(axis)[0])

        # round epoch in successive units for large scales
        unit = self.get_unit()
        date = from_gps(epoch)
        fields = ("second", "minute", "hour", "day")
        for i, u in enumerate(fields[1:]):
            if unit < units.Unit(u):
                break
            if u in ("day",):
                date = date.replace(**{fields[i]: 1})
            else:
                date = date.replace(**{fields[i]: 0})
        return int(to_gps(date))

    def _auto_unit(self, axis: Axis) -> UnitBase:
        """Find the best scaled unit for this ``axis``."""
        # get width of axis
        vmin, vmax = self._lim(axis)
        duration = vmax - vmin

        # find time unit that fits the duration well;
        # the magic scaling of 4 or 0.01 is entirely arbitrary,
        # but in practice results in figures that scale nicely
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

        # if nothing else worked, just use seconds
        return units.second

    def get_transform(self) -> GPSTransform:
        """Return the `GPSTransform` associated with this scale."""
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
            return self.Transform(
                unit=self.get_unit(),
                epoch=self.get_epoch(),
            )
        finally:  # reset to current settings
            self.set_epoch(epoch)
            self.set_unit(unit)


# -- registrations -------------------

def register_gps_scale(scale_class: type[GPSScale]) -> None:
    """Register a new GPS scale.

    ``scale_class`` must be a subclass of `GPSScale`.
    """
    register_scale(scale_class)
    GPS_SCALES[scale_class.name] = scale_class


def _gps_scale_factory(unit: UnitBase) -> type[GPSScale]:
    """Construct a GPSScale for this unit."""

    class FixedGPSScale(GPSScale):
        """`GPSScale` for a specific GPS time unit."""

        name = (unit.long_names or unit.names)[0] + "s"

        def __init__(
            self,
            axis: Axis,
            epoch: Number | Decimal | SupportsToGps | None = None,
        ) -> None:
            super().__init__(axis, epoch=epoch, unit=unit)

    return FixedGPSScale


register_gps_scale(GPSScale)  # auto-gps

for _unit in TIME_UNITS:
    # don't go past 'year' for GPSScale
    if _unit is units.kiloyear:
        break
    register_gps_scale(_gps_scale_factory(_unit))

# update the docstring for matplotlib scale methods
with contextlib.suppress(AttributeError):
    _docstring.interpd.params.update(
        scale=" | ".join([repr(x) for x in get_scale_names()]),
        scale_docs=get_scale_docs().rstrip(),
    )
