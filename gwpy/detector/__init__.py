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

"""Defines each LaserInterferometer detector in the current network
"""

from .interferometers import *
from .channel import *

# build list of known detectors from LAL
try:
    from lal import lalCachedDetectors
except ImportError:
    pass
else:
    __all__ = ([ifo.frDetector.name for ifo in lalCachedDetectors] +
               ["DETECTOR_BY_PREFIX"])
    DETECTOR_BY_PREFIX = dict()
    for ifo in lalCachedDetectors:
        detector = LaserInterferometer()
        detector.prefix = ifo.frDetector.prefix
        detector.name = ifo.frDetector.name
        detector.vertex = ifo.location
        detector.response_matrix = ifo.response
        globals()[detector.name] = detector
        DETECTOR_BY_PREFIX[detector.prefix] = detector
    del lalCachedDetectors
