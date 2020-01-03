# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Text formatting for GWpy plots
"""

from matplotlib import rcParams

from astropy import units

from . import tex

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def to_string(input_):
    """Format an input for representation as text

    This method is just a convenience that handles default LaTeX formatting
    """
    usetex = rcParams['text.usetex']
    if isinstance(input_, units.UnitBase):
        return input_.to_string('latex_inline')
    if isinstance(input_, (float, int)) and usetex:
        return tex.float_to_latex(input_)
    if usetex:
        return tex.label_to_latex(input_)
    return str(input_)


def default_unit_label(axis, unit):
    """Set default label for an axis from a `~astropy.units.Unit`

    If the axis already has a label, this function does nothing.

    Parameters
    ----------
    axis : `~matplotlib.axis.Axis`
        the axis to manipulate

    unit : `~astropy.units.Unit`
        the unit to use for the label

    Returns
    -------
    text : `str`, `None`
        the text for the new label, if set, otherwise `None`
    """
    if not axis.isDefault_label:
        return
    label = axis.set_label_text(unit.to_string('latex_inline_dimensional'))
    axis.isDefault_label = True
    return label.get_text()
