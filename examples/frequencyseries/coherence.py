#!/usr/bin/env python

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

"""Calculating the coherence between two channels

The `coherence <http://en.wikipedia.org/wiki/Coherence_(physics)>`_ between
two channels is a measure of the frequency-domain correlation between their
time-series data.

In LIGO, the coherence is a crucial indicator of how noise sources couple into
the main differential arm-length readout.

In this example we calculate the coherence between two length-sensing
degrees of freedom, the Signal-Recycling Cavity length (SRCL), and the
Common-Arm motion (CARM).

"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we import the `TimeSeriesDict`
from gwpy.timeseries import TimeSeriesDict

# and then :meth:`~TimeSeriesDict.get` both data sets:
data = TimeSeriesDict.get(['L1:LSC-SRCL_IN1_DQ', 'L1:LSC-CARM_IN1_DQ'],
                           'Feb 13 2015', 'Feb 13 2015 00:15')

# We can then calculate the :meth:`~TimeSeries.coherence` of one
# `TimeSeries` with respect to the other, using an 8-second Fourier
# transform length, with a 4-second (50%) overlap:
coh = data['L1:LSC-SRCL_IN1_DQ'].coherence(data['L1:LSC-CARM_IN1_DQ'], 8, 4)

# Finally, we can :meth:`~gwpy.frequencyseries.FrequencySeries.plot` the resulting data:
plot = coh.plot(figsize=[12, 6], label=None)
ax = plot.gca()
ax.set_yscale('linear')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Coherence')
ax.set_title('Coherence between SRCL and CARM for L1')
ax.grid(True, 'both', 'both')
plot.show()
