#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019-2020)
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

"""Generating an inspiral range timeseries

One standard figure-of-merit for the sensitivity of a gravitational-wave
detector is the distance to which a binary neutron star (BNS) inspiral
with two 1.4 solar mass components would be detected with a signal-to-noise
ratio (SNR) of 8. We can estimate this using
:func:`gwpy.astro.range_timeseries` directly from the strain readout for
a detector.
"""

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Alex Urban <alexander.urban@ligo.org>'

# First, we need to load some data. We can `fetch` the
# `public data <https://www.gw-openscience.org/catalog/>`__
# around the GW170817 BNS merger:

from gwpy.timeseries import TimeSeries
h1 = TimeSeries.fetch_open_data('H1', 1187006834, 1187010930)
l1 = TimeSeries.fetch_open_data('L1', 1187006834, 1187010930)

# Then, we can measure the inspiral range directly:

from gwpy.astro import range_timeseries
h1range = range_timeseries(h1, 30, fftlength=4, fmin=10)
l1range = range_timeseries(l1, 30, fftlength=4, fmin=10)

# We can now plot these trends to see the variation in LIGO
# sensitivity over an hour or so surrounding GW170817:

plot = h1range.plot(
    label='LIGO-Hanford', color='gwpy:ligo-hanford', figsize=(12, 5))
ax = plot.gca()
ax.plot(l1range, label='LIGO-Livingston', color='gwpy:ligo-livingston')
ax.set_ylabel('Angle-averaged sensitive distance [Mpc]')
ax.set_title('LIGO sensitivity to BNS around GW170817')
ax.set_epoch(1187008882)  # <- set 0 on plot to GW170817
ax.legend()
plot.show()

# Note, the extreme dip in LIGO-Livingston's sensitivity near GW170817
# is caused by a loud, transient noise event, see `Phys. Rev. Lett.
# vol. 119, p. 161101 <http://doi.org/10.1103/PhysRevLett.119.161101>`_
# for more information.
