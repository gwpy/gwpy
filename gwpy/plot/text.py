# Copyright (c) 2017-2025 Cardiff University
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

"""Text formatting for GWpy plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astropy.units import UnitBase
from matplotlib import rcParams

from . import tex

if TYPE_CHECKING:
    from matplotlib.axis import Axis

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def to_string(
    input_: str | float | UnitBase,
) -> str:
    """Format an input for representation as text.

    This method is just a convenience that handles default LaTeX formatting.
    """
    usetex = rcParams["text.usetex"]
    if isinstance(input_, UnitBase):
        return input_.to_string("latex_inline")
    if isinstance(input_, float | int) and usetex:
        return tex.float_to_latex(input_)
    if usetex:
        return tex.label_to_latex(input_)
    return str(input_)


def default_unit_label(
    axis: Axis,
    unit: UnitBase,
    format: str = "latex_inline_dimensional",  # noqa: A002
) -> str:
    """Set default label for an axis from a `~astropy.units.Unit`.

    If the axis already has a label, this function does nothing except
    return the axis label.

    Parameters
    ----------
    axis : `~matplotlib.axis.Axis`
        The axis to manipulate.

    unit : `~astropy.units.Unit`
        The unit to use for the label.

    format : `str`, optional
        The format to use when converting the unit to a label.

    Returns
    -------
    text : `str`
        The text for the new label, if set.
    """
    if axis.isDefault_label:
        axis.set_label_text(unit.to_string(format=format))
        axis.isDefault_label = True
    return axis.get_label_text()
