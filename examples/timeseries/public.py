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

"""Plotting public LIGO data

I would like to study the gravitational wave strain time-series around the time of an interesting simulated signal during the last science run (S6).

These data are public, so we can load them directly from the web.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First: import everything we need (and nothing we don't need)
from urllib2 import urlopen
from numpy import asarray
from gwpy.timeseries import TimeSeries

# Next, download the data as a string of text
data = urlopen('http://www.ligo.org/science/GW100916/L-strain_hp30-968654552-10.txt').read()

# We can now parse the text as a list of floats, and generate a `TimeSeries`
# by supplying the necessary metadata
ts = TimeSeries(asarray(data.splitlines(), dtype=float),
                epoch=968654552, sample_rate=16384, unit='strain')

# Finally, we can make a plot:
plot = ts.plot()
plot.set_title('LIGO Livingston Observatory data for GW100916')
plot.set_ylabel('Gravitational-wave strain amplitude')
plot.show()
