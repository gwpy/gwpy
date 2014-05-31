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

"""GWpy Example: comparing a the Spectrum of a channel at different times

Problem
-------

I'm interested in comparing the amplitude spectrum of a channel between a
known 'good' time - where the spectrum is what we expect it to be - and a
known 'bad' time - where some excess noise appeared and the spectrum
changed appreciably.

"""

from gwpy.timeseries import TimeSeries

from gwpy import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

# set the times
goodtime = 1061800700
badtime = 1061524816
duration = 120

# read the data over the network
gooddata = TimeSeries.fetch('L1:PSL-ISS_PDB_OUT_DQ', goodtime, goodtime+duration)
gooddata.name = 'Clean data'
baddata = TimeSeries.fetch('L1:PSL-ISS_PDB_OUT_DQ', badtime, badtime+duration)
baddata.name = 'Noisy data'

# calculate spectrum with 1/8 Hz resolution
goodasd = gooddata.asd(4, 2)
badasd = baddata.asd(4, 2)

# plot
plot = badasd.plot()
ax = plot.gca()
ax.plot(goodasd)
ax.set_xlim(10, 8000)
ax.set_ylim(1e-6, 5e-4)

if __name__ == '__main__':
    try:
        outfile = __file__.replace('.py', '.png')
    except NameError:
        pass
    else:
        plot.save(outfile)
        print("Example output saved as\n%s" % outfile)
