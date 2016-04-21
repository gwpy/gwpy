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
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>

"""Calculating and plotting a `SpectralVariance` histogram
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.frequencyseries'

# In order to generate a `SpectralVariance` histogram we need to import the
# `~gwpy.timeseries.TimeSeries` and :meth:`~gwpy.timeseries.TimeSeries.get`
# the data:
from gwpy.timeseries import TimeSeries
llo = TimeSeries.get(
    'L1:LDAS-STRAIN,rds', 'August 1 2010', 'August 1 2010 00:10')

# We can then call the :meth:`~gwpy.timeseries.TimeSeries.spectral_variance`
# method of the ``llo`` `~gwpy.timeseries.TimeSeries`:
variance = llo.spectral_variance(1, log=True, low=1e-24, high=1e-19, nbins=100)

# We can then :meth:`~SpectralVariance.plot` the `SpectralVariance`
plot = variance.plot(norm='log', vmin=0.5, vmax=100)
ax = plot.gca()
ax.grid()
ax.set_xlim(40, 4096)
ax.set_ylim(1e-24, 1e-19)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'GW ASD [strain/\rtHz]')
ax.set_title('LIGO Livingston Observatory sensitivity variance')
plot.show()
