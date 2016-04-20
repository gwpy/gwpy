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

"""Calculating and plotting a `FrequencySeries`

I'm interested in the level of ground motion surrounding a particular time
during commissioning of the Advanced LIGO Livingston Observatory. I don't
have access to the frame files on disk, so I'll need to use NDS.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.frequencyseries'

# In order to generate a `FrequencySeries` we need to import the
# `~gwpy.timeseries.TimeSeries` and :meth:`~gwpy.timeseries.TimeSeries.get`
# the data:
from gwpy.timeseries import TimeSeries
lho = TimeSeries.get(
    'H1:LDAS-STRAIN,rds', 'August 1 2010', 'August 1 2010 00:02')
llo = TimeSeries.get(
    'L1:LDAS-STRAIN,rds', 'August 1 2010', 'August 1 2010 00:02')

# We can then call the :meth:`~gwpy.timeseries.TimeSeries.asd` method to
# calculated the amplitude spectral density for each
# `~gwpy.timeseries.TimeSeries`:
lhoasd = lho.asd(2, 1)
lloasd = llo.asd(2, 1)

# We can then :meth:`~FrequencySeries.plot` the spectra
plot = lhoasd.plot(color='b', label='LHO')
ax = plot.gca()
ax.plot(lloasd, color='g', label='LLO')
ax.set_xlim(40, 4096)
ax.set_ylim(1e-23, 7.5e-21)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_right()
plot.show()
