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

"""Plotting a transfer function

I would like to study how a signal transfers from one part of the
interferometer to another.

Specifically, it is interesting to measure the amplitude transfer of
ground motion through the HEPI system.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

if __name__ == '__main__':
    from matplotlib import pyplot
    pyplot.ion()

# Before anything else, we import the objects we will need:
from gwpy.time import tconvert
from gwpy.timeseries import TimeSeriesDict
from gwpy.plot import BodePlot

# and set the times of our query, and the channels we want:
start = tconvert('May 27 2014 04:00')
end = start + 1800
gndchannel = 'L1:ISI-GND_STS_ITMY_Z_DQ'
hpichannel = 'L1:HPI-ITMY_BLND_L4C_Z_IN1_DQ'

# We can call the :meth:`~TimeSeriesDict.get` method of the `TimeSeriesDict`
# to retrieve all data in a single operation:
data = TimeSeriesDict.get([gndchannel, hpichannel], start, end, verbose=True)
gnd = data[gndchannel]
hpi = data[hpichannel]

# The transfer function between time series is easily computed with the
# :meth:`~TimeSeries.transfer_function` method:
tf = gnd.transfer_function(hpi, 100, 50)

# The `~gwpy.plot.BodePlot` knows how to separate a complex-valued
# `~gwpy.frequencyseries.FrequencySeries` into magnitude and phase:
plot = BodePlot(tf)
plot.maxes.set_title(
    r'L1 ITMY ground $\rightarrow$ HPI transfer function')
plot.maxes.set_ylim(-55, 50)
plot.show()
