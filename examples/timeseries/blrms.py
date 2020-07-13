#!/usr/bin/env python
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

"""Comparing seismic trends between LIGO sites

On Feb 13 2015 there was a massive earthquake in the Atlantic Ocean, that
should have had an impact on LIGO operations, I'd like to find out.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First: we import the objects we need, one for getting the data:
from gwpy.timeseries import TimeSeriesDict
# and one for plotting the data:
from gwpy.plot import Plot

# Next we define the channels we want, namely the 0.03Hz-1Hz ground motion
# band-limited RMS channels (1-second average trends).
# We do this using string-replacement so we can substitute the interferometer
# prefix easily when we need to:
channels = [
    '{ifo}:ISI-BS_ST1_SENSCOR_GND_STS_X_BLRMS_30M_100M.mean,s-trend',
    '{ifo}:ISI-BS_ST1_SENSCOR_GND_STS_Y_BLRMS_30M_100M.mean,s-trend',
    '{ifo}:ISI-BS_ST1_SENSCOR_GND_STS_Z_BLRMS_30M_100M.mean,s-trend',
]

# At last we can :meth:`~TimeSeriesDict.get` 12 hours of data for each
# interferometer:
lho = TimeSeriesDict.get([c.format(ifo='H1') for c in channels],
                         'Feb 13 2015 16:00', 'Feb 14 2015 04:00')
llo = TimeSeriesDict.get([c.format(ifo='L1') for c in channels],
                         'Feb 13 2015 16:00', 'Feb 14 2015 04:00')

# Next we can plot the data, with a separate `~gwpy.plot.Axes` for each
# instrument:
plot = Plot(lho, llo, figsize=(12, 6), sharex=True, yscale='log')
ax1, ax2 = plot.axes
for ifo, ax in zip(('Hanford', 'Livingston'), (ax1, ax2)):
    ax.legend(['X', 'Y', 'Z'])
    ax.text(1.01, 0.5, ifo, ha='left', va='center', transform=ax.transAxes,
            fontsize=18)
ax1.set_ylabel(r'$1-3$\,Hz motion [nm/s]', y=-0.1)
ax2.set_ylabel('')
ax1.set_title('Magnitude 7.1 earthquake impact on LIGO')
plot.show()

# As we can see, the earthquake had a huge impact on the LIGO observatories,
# severly imparing operations for several hours.
