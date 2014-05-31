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

"""GWpy Example: plotting a spectrogram

Problem
-------

I would like to study the gravitational wave strain spectrogram around the time of an interesting simulated signal during the last science run (S6).
"""

from gwpy import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

# import TimeSeries class
from gwpy.timeseries import TimeSeries

# find the data using NDS
gwdata = TimeSeries.fetch('H1:LDAS-STRAIN', 'September 16 2010 06:40', 'September 16 2010 06:50')
gwdata.unit = 'strain'

# calculate spectrogram
specgram = gwdata.spectrogram(5, fftlength=2, overlap=1) ** (1/2.)

# make plot and add colour scale
plot = specgram.plot(norm='log', vmin=1e-23, vmax=1e-19)
plot.set_ylim(40, 4000)
plot.add_colorbar(label=r'Gravitational-wave amplitude [strain/$\sqrt{\mathrm{Hz}}$]')

if __name__ == '__main__':
    try:
        outfile = __file__.replace('.py', '.png')
    except NameError:
        pass
    else:
        plot.save(outfile)
        print("Example output saved as\n%s" % outfile)
