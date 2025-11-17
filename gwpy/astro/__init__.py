# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Methods for sensitivity calculations of gravitational-wave interferometer data.

The LIGO project measures time-dependent sensitivity by calculating the
distance at which the gravitational-wave signature of a binary neutron star
(BNS) inspiral would be recorded by an instrument with a signal-to-noise ratio
(SNR) of 8.
In most of the literature, this is known as the 'inspiral range' or the
'horizon distance'.
"""

from .range import (
    burst_range,
    burst_range_spectrum,
    inspiral_range,
    inspiral_range_psd,
    range_spectrogram,
    range_timeseries,
    sensemon_range,
    sensemon_range_psd,
)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
