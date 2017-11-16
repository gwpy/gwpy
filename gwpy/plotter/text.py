# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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
    if isinstance(input_, (float, int)) and usetex:
        return tex.float_to_latex(input_)
    elif usetex:
        return tex.label_to_latex(input_)
    return str(input_)


def unit_as_label(unit):
    # pylint: disable=anomalous-backslash-in-string
    """Format a unit as a label for an Axis

    Parameters
    ----------
    unit : `~astropy.units.UnitBase`
        the input unit to format

    Returns
    -------
    ustr : `str`
        a string representation of the unit, as a label

    Examples
    --------
    >>> unit_as_label(Unit('Hertz'))
    'Frequency [Hz]'
    >>> unit_as_label(Unit('m / s'))
    '[\mathrm{m}/\mathrm{s}]'
    """
    if rcParams['text.usetex']:
        ustr = tex.unit_to_latex(unit)
    elif isinstance(unit, units.UnitBase):
        ustr = unit.to_string()
    if unit.physical_type and unit.physical_type != 'unknown':
        return '%s [%s]' % (to_string(unit.physical_type.title()), ustr)
    return '[%s]' % ustr
