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

"""Support for plotting with units
"""

from astropy.units.format import LatexInline


class LatexInlineDimensional(LatexInline):
    r"""Custom LaTeX formatter that includes physical type (if available)

    Mainly for auto-labelling `Axes` in matplotlib figures.

    Examples
    --------
    The built-in astropy ``latex_inline`` formatter gives this:

    >>> Unit('m/s').to_string(format='latex_inline')
    '$\mathrm{m\,s^{-1}}$'

    This custom 'dimensional' formatter gives:

    >>> Unit('m/s').to_string(format='latex_inline_dimensional')
    'Speed [$\mathrm{m\,s^{-1}}$]'
    """
    name = 'latex_inline_dimensional'

    @classmethod
    def to_string(cls, unit, *args, **kwargs):
        u = f"[{super().to_string(unit, *args, **kwargs)}]"

        if unit.physical_type not in {None, 'unknown', 'dimensionless'}:
            # format physical type of unit for LaTeX
            ptype = str(unit.physical_type).title().replace("_", r"\_")
            # return '<Physical type> [<unit>]'
            return f"{ptype} {u}"
        return u
