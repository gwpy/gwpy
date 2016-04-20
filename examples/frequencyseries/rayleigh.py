#!/usr/bin/env python

# Copyright (C) Duncan Macleod (2013-2015)
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

"""Plotting a Rayleigh-statistic `Spectum`

"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.frequencyseries'

# First, we import the :class:`~gwpy.timeseries.TimeSeries` and :meth:`~gwpy.timeseries.TimeSeries.get` the data:
from gwpy.timeseries import TimeSeries
gwdata = TimeSeries.get(
    'H1:LDAS-STRAIN,rds', 'September 16 2010 06:40', 'September 16 2010 06:50')

# Next, we can calculate a Rayleigh statistic `FrequencySeries` using the 
# :meth:`~gwpy.timeseries.TimeSeries.rayleigh_spectrum` method of the
# `~gwpy.timeseries.TimeSeries` with a 2-second FFT and 1-second overlap (50%):
rayleigh = gwdata.rayleigh_spectrum(2, 1)

# and can make a plot using the :meth:`~FrequencySeries.plot` method
plot = rayleigh.plot()
plot.set_xscale('log')
plot.set_xlim(40, 4000)
plot.set_xlabel('Frequency [Hz]')
plot.set_ylabel('Rayleigh statistic')
plot.show()
