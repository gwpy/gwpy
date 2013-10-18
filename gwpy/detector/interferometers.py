# Copyright (C) Duncan Macleod (2013)
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

"""Defines objects representing the laser interferometer GW detector
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

import lal


class LaserInterferometer(object):

    def __init__(self):
        self.name = None
        self.prefix = None
        self.location = None
        self.x_end_location = None
        self.y_end_location = None
        self.response_matrix = None

    def response(self, coord, polarization=0.0):
        """Determine the F+, Fx antenna responses to a signal
        originating at the given coordinates.
        """
        return lal.ComputeDetAMResponse(
                   self.response_matrix, coord.ra.radians, coord.dec.radians,
                   polarization,
                   lal.GreenwichMeanSiderealTime(coord.obstime.gps))

    def time_delay(self, other, coord):
        return lal.ArrivalTimeDiff(self.response_matrix, other.response,
                                   coord.ra.radians, coord.dec.radians,
                                   coord.obstime.gps)
