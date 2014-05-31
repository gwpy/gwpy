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
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>

"""GWpy Example: plotting a spectrum

Problem
-------

I'm interested in the level of ground motion surrounding a particular time
during commissioning of the Advanced LIGO Livingston Observatory. I don't
have access to the frame files on disk, so I'll need to use NDS.

"""

from gwpy.time import tconvert
from gwpy.timeseries import TimeSeries

from gwpy import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

# read the data over the network
lho = TimeSeries.fetch('H1:LDAS-STRAIN', 'August 1 2010', 'August 1 2010 00:02')
lho.unit = 'strain'
#llo = TimeSeries.fetch('L1:LDAS-STRAIN', start.gps, end.gps, verbose=True)
#llo.unit = 'strain'

# calculate spectrum with 0.5Hz resolution
lhoasd = lho.asd(2, 1)
#lloasd = llo.asd(2, 1)

# plot
plot = lhoasd.plot(color='b', label='LHO')
#plot.add_spectrum(lloasd, color='g', label='LLO')
plot.xlim = [40, 4096]
plot.ylim = [1e-23, 7.5e-21]

# hide some axes
plot.axes[0].spines['right'].set_visible(False)
plot.axes[0].spines['top'].set_visible(False)
plot.axes[0].get_xaxis().tick_bottom()
plot.axes[0].get_yaxis().tick_right()

if __name__ == '__main__':
    try:
        outfile = __file__.replace('.py', '.png')
    except NameError:
        pass
    else:
        plot.save(outfile)
        print("Example output saved as\n%s" % outfile)
