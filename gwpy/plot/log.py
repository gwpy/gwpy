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

"""Utilities for plotting on logarithmically-scaled axes."""

from __future__ import annotations

from functools import wraps
from math import (
    ceil,
    floor,
    isclose,
    log,
)
from typing import TYPE_CHECKING

import numpy
from matplotlib import (
    rcParams,
    ticker as mticker,
)
from matplotlib.scale import (
    LogScale as _LogScale,
    register_scale,
)

if TYPE_CHECKING:
    from matplotlib.axis import Axis
    from numpy.typing import NDArray

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def _math(s: str | float) -> str:
    """Format a number using mathtext or TeX."""
    if rcParams["text.usetex"]:
        return f"${s}$"
    return fr"$\mathdefault{{{s}}}$"


def _render_simple(
    values: NDArray,
    ndec: int = 2,
    thresh: float = 100001,
) -> bool:
    """Return `True` if all of ``values`` can be rendered as simple floats."""
    return (
        # is not empty
        bool(values.size)
        # and not all zeros
        and not all(values == 0)
        # and max below very large number
        and values.max() < thresh
        # and all <2 dec. places
        and numpy.array_equal(values, values.round(ndec))
    )


class LogFormatter(mticker.LogFormatterMathtext):
    """Format values for log axis.

    This `LogFormatter` extends the standard
    `~matplotlib.ticker.LogFormatterMathtext` to print numbers in the
    range [0.01, 1000) normally, and all others via the standard
    `~matplotlib.ticker.LogFormatterMathtext` output.
    """

    @wraps(mticker.LogFormatterMathtext.format_ticks)
    def format_ticks(self, values: list[float]) -> list[str]:
        """Format the set of ticks."""
        # this method overrides the default to enable formatting ticks
        # using simple float/integer representations (as opposed) to
        # scientific notation, if _all_ of the ticks have a value
        # small enough to render nicely (roughly <=1000 for visible ticks)
        # and no more than two decimal places
        self.set_locs(values)

        # remove floating-point precision errors
        values2 = numpy.asanyarray([float(f"{x:.12g}") for x in values])

        # if can render using just "%s" do it
        if _render_simple(values2):
            return [
                self(int(x) if x.is_integer() else x, pos=i, fmt="%s")
                for i, x in enumerate(values2)
            ]
        # otherwise use the matplotlib default
        return super().format_ticks(values)

    def _num_ticks(self) -> int:
        """Return the number of ticks that should be displayed."""
        if self.axis is None:
            return 0
        viewlim = self.axis.get_view_interval()
        loglim = numpy.log10(viewlim)
        return numpy.arange(
            ceil(loglim[0]),
            floor(ceil(loglim[1])),
            dtype=int,
        ).size

    @wraps(mticker.LogFormatterMathtext.set_locs)
    def set_locs(self, locs: list[float] | None = None) -> None:
        """Set the locations of the ticks."""
        ret = super().set_locs(locs=locs)

        # if a single major tick, but matplotlib decided not to include
        # sub-ticks, then there is more than a decade on the axis, so
        # (for base 10) include half-decade ticks
        if (
                self._num_ticks() == 1
                and self._sublabels == {1}
                and self._base == 10  # noqa: PLR2004
        ):
            self._sublabels = {1., 5., 10.}

        return ret

    @wraps(mticker.LogFormatterMathtext.__call__)
    def __call__(
        self,
        x: float,
        pos: int | None = None,
        fmt: str | None = None,
    ) -> str:
        """Format a tick label."""
        if x == 0:
            return _math((fmt or "%s") % 0)

        # work out whether to show this label
        # if there are enough major ticks or this formatter doesn't support
        # minor ticks, return a blank string
        x = abs(x)
        b = self._base
        fx = log(x) / log(b)
        is_x_decade = isclose(fx, round(fx))
        exponent = numpy.round(fx) if is_x_decade else numpy.floor(fx)
        coeff = numpy.round(x / b ** exponent)
        nticks = self._num_ticks()
        if (
                nticks >= 1
                and self._sublabels is not None
                and coeff not in self._sublabels
        ):
            return ""

        # enable custom format
        if fmt:
            sign = "-" if x < 0 else ""
            return _math(sign + fmt % x)

        return super().__call__(x, pos=pos)


class LogScale(_LogScale):
    """GWpy version of the matplotlib `LogScale`.

    This scale overrides the default to use the new GWpy formatters
    for major and minor ticks.
    """

    @wraps(_LogScale.set_default_locators_and_formatters)
    def set_default_locators_and_formatters(self, axis: Axis) -> None:
        """Set the default locators and formatters."""
        axis.set_major_locator(
            mticker.LogLocator(self.base),
        )
        axis.set_major_formatter(
            LogFormatter(self.base),
        )
        axis.set_minor_locator(
            mticker.LogLocator(self.base, self.subs),
        )
        axis.set_minor_formatter(
            LogFormatter(self.base, labelOnlyBase=bool(self.subs is not None)),
        )


register_scale(LogScale)
