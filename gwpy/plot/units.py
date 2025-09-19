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

"""Support for plotting with units."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from astropy.units.format import LatexInline

if TYPE_CHECKING:
    from astropy.units import UnitBase


class LatexInlineDimensional(LatexInline):
    r"""Custom LaTeX formatter that includes physical type (if available).

    Mainly for auto-labelling `Axes` in matplotlib figures.

    Examples
    --------
    The built-in astropy ``latex_inline`` formatter gives this:

    >>> Unit('m/s').to_string(format='latex_inline')
    '$\mathrm{m\,s^{-1}}$'

    This custom 'dimensional' formatter gives:

    >>> Unit('m/s').to_string(format='latex_inline_dimensional')
    'Speed/Velocity [$\mathrm{m\,s^{-1}}$]'
    """

    name = "latex_inline_dimensional"

    @classmethod
    @wraps(LatexInline.to_string)
    def to_string(
        cls,
        unit: UnitBase,
        **kwargs,
    ) -> str:
        """Output this unit in LaTeX format with dimensions."""
        if unit.physical_type == "dimensionless":
            return "Dimensionless"

        u = f"[{super().to_string(unit, **kwargs)}]"

        if (
            unit.physical_type is None
            or unit.physical_type == "unknown"
        ):
            return u

        # format physical type of unit for LaTeX
        ptype = str(unit.physical_type).title().replace("_", r"\_")
        # looks like '<Physical type> [<unit>]'
        return f"{ptype} {u}"
