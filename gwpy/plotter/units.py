#!/usr/bin/env python

# Copyright (C) 2012 Duncan M. Macleod
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Docstring
"""

from matplotlib import units

from .ticks import GPSFormatter

from lal import git_version as version
from lal.lal import LIGOTimeGPS

__author__ = "Duncan M. Macleod <duncan.macleod@ligo.org>"
__version__ = version.id
__date__ = version.date


class GPSConverter(units.ConversionInterface):
    """Use GPS units in plot
    """
    def convert(value, unit, axis):
        return value

    def axisinfo(unit, axis):
        if unit != "gps":
            return None
        formatter = GPSFormatter()
        return units.AxisInfo(majfmt=formatter, label="gps")

    def default_units(x, axis):
        return "gps"

units.registry[LIGOTimeGPS] = GPSConverter()
