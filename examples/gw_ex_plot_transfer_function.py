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
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>"""GWpy Example: plotting a time-series

"""
GWpy Example: plotting a transfer function
------------------------------------------

I would like to study how a signal transfers from one part of the
interferometer to another.

Specifically, it is interesting to measure the amplitude transfer of
ground motion through the HEPI system.
"""

from gwpy.time import tconvert
from gwpy.timeseries import TimeSeriesDict
from gwpy.plotter import BodePlot

from gwpy import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

# set the times
start = tconvert('May 27 2014 04:00')
end = start + 1800

gndchannel = 'L1:ISI-GND_STS_ITMY_Z_DQ'
hpichannel = 'L1:HPI-ITMY_BLND_L4C_Z_IN1_DQ'

# get data
data = TimeSeriesDict.fetch([gndchannel, hpichannel], start, end, verbose=True)
gnd = data[gndchannel]
gnd.name = 'Before HEPI (ground)'
hpi = data[hpichannel]
hpi.name = 'After HEPI'

gnd.unit = 'nm/s'
hpi.unit = 'nm/s'

# get FFTs
gndfft = gnd.average_fft(100, 50, window='hamming')
hpifft = hpi.average_fft(100, 50, window='hamming')

# get transfer function (up to lower Nyquist)
size = min(gndfft.size, hpifft.size)
tf = hpifft[:size] / gndfft[:size]

plot = BodePlot(tf)
magax = plot.axes[0]
magax.set_title(r'L1 ITMY ground $\rightarrow$ HPI transfer function')
magax.set_ylim(-55, 50)


if __name__ == '__main__':
    try:
        outfile = __file__.replace('.py', '.png')
    except NameError:
        pass
    else:
        plot.save(outfile)
        print("Example output saved as\n%s" % outfile)
