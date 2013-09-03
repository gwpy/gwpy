#!/usr/bin/env python

# Copyright (C) 2012 Duncan Macleod
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

"""Defines GPS time format, if astropy doesn't supply it
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

try:
    from astropy.time import TimeGPS
except ImportError:
    from astropy.time.core import TimeFromEpoch,SECS_PER_DAY,TIME_FORMATS

    class TimeGPS(TimeFromEpoch):
        """GPS time format: seconds from 1980-01-06 00:00:00 UTC.
        """
        name = 'gps'
        precision = 9
        unit = 1 / float(SECS_PER_DAY)
        epoch_val = '1980-01-06 00:00:19'
        epoch_val2 = None
        epoch_scale = 'tai'
        epoch_format = 'iso'

    TIME_FORMATS[TimeGPS.name] = TimeGPS
