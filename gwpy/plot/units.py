# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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
    """Custom LaTeX formatter that includes physical type (if available)

    Mainly for auto-labelling `Axes` in matplotlib figures
    """
    name = 'latex_inline_dimensional'

    @classmethod
    def to_string(cls, unit):
        u = '[{0}]'.format(super().to_string(unit))

        if unit.physical_type not in {None, 'unknown', 'dimensionless'}:
            ptype = str(unit.physical_type).split('/', 1)[0].title()
            return '{0} {1}'.format(cls._latex_escape(ptype), u)
        return u
