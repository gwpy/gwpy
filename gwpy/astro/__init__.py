# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""The `astro` module provides methods for sensitivity calculations of
gravitational-wave interferometer data.

The LIGO project measures time-dependent sensitivity by calculating the
distance at which the gravitational-wave signature of a binary neutron star
(BNS) inspiral would be recorded by an instrument with a signal-to-noise ratio
(SNR) of 8.
In most of the literature, this is known as the 'inspiral range' or the
'horizon distance'.

The following methods are provided in order to calculate the sensitive
distance range of a detector

.. autosummary::

   ~gwpy.astro.burst_range
   ~gwpy.astro.burst_range_spectrum
   ~gwpy.astro.inspiral_range
   ~gwpy.astro.inspiral_range_psd
   ~gwpy.astro.sensemon_range
   ~gwpy.astro.sensemon_range_psd
   ~gwpy.astro.range_timeseries
   ~gwpy.astro.range_spectrogram

Each of the above methods has been given default parameters corresponding to
the standard usage by the LIGO project.
"""

from .range import (
    burst_range,
    burst_range_spectrum,
    inspiral_range,
    inspiral_range_psd,
    sensemon_range,
    sensemon_range_psd,
    range_timeseries,
    range_spectrogram,
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
