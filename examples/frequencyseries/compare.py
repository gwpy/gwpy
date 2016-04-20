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

"""Comparing the same `Channel` at different times

I'm interested in comparing the amplitude spectrum of a channel between a
known 'good' time - where the spectrum is what we expect it to be - and a
known 'bad' time - where some excess noise appeared and the spectrum
changed appreciably.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we import the `TimeSeries`
from gwpy.timeseries import TimeSeries

# And we set the times of our investigation:
goodtime = 1061800700
badtime = 1061524816
duration = 120

# Next we :meth:`~TimeSeries.get` the data:
gooddata = TimeSeries.get('L1:PSL-ISS_PDB_OUT_DQ', goodtime, goodtime+duration)
baddata = TimeSeries.get('L1:PSL-ISS_PDB_OUT_DQ', badtime, badtime+duration)

# and calculate an `amplitude spectral density (ASD) <TimeSeries.asd>` using a 4-second Fourier transform with a 2-second overlap:
goodasd = gooddata.asd(4, 2)
badasd = baddata.asd(4, 2)

# Lastly, we make a plot of the data by `plotting <FrequencySeries.plot>` one `~gwpy.frequencyseries.FrequencySeries`, and then adding the second:
plot = badasd.plot(label='Noisy data')
ax = plot.gca()
ax.plot(goodasd, label='Clean data')
ax.set_xlabel('Frequency [Hz]')
ax.set_xlim(10, 8000)
ax.set_ylabel(r'Noise ASD [1/$\sqrt{\mathrm{Hz}}$]')
ax.set_ylim(1e-6, 5e-4)
ax.grid(True, 'both', 'both')
plot.show()
