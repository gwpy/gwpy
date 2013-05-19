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

"""Defines each LaserInterferometer detector in the current network
"""

from .interferometers import LaserInterferometer
from ..utils import lal

from .channel import Channel

if lal.SWIG_LAL:
    swiglal = lal.swiglal
    __all__ = ([ifo.frDetector.name for ifo in swiglal.lalCachedDetectors] + 
               ["DETECTOR_BY_PREFIX"])

    DETECTOR_BY_PREFIX = dict()
    
    for ifo in swiglal.lalCachedDetectors:
        detector = LaserInterferometer()
        detector.prefix = ifo.frDetector.prefix
        detector.name = ifo.frDetector.name
        detector.location = ifo.location
        detector.response_matrix = ifo.response
        globals()[detector.name] = detector
        DETECTOR_BY_PREFIX[detector.prefix] = detector
