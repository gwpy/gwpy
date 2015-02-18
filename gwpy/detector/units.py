# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013-2015)
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

"""This module registers a number of custom units used in GW astronomy.
"""

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from astropy import units

# enable imperial units
units.add_enabled_units(units.imperial)

# -----------------------------------------------------------------------------
# instrumental units

units.add_enabled_units([
    units.def_unit(['counts'], represents=units.Unit('count')),
    units.def_unit(['undef'], doc='No unit has been defined for these data'),
    units.def_unit(['coherence'], represents=units.dimensionless_unscaled),
    units.def_unit(['strain'], represents=units.dimensionless_unscaled),
    units.def_unit(['Degrees_C'], represents=units.Unit('Celsius')),
    units.def_unit(['Degrees_F'], represents=units.Unit('Fahrenheit')),
])
