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
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>

"""Calculating and plotting a `FrequencySeries`

The LIGO Laboratory has publicly released the strain data around the time of
the GW150914 gravitational-wave detection; we can use these to calculate
and display the spectral sensitivity of each of the detectors at that time.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.frequencyseries'

# In order to generate a `FrequencySeries` we need to import the
# `~gwpy.timeseries.TimeSeries` and use
# :meth:`~gwpy.timeseries.TimeSeries.fetch_open_data` to download the strain
# records:

from gwpy.timeseries import TimeSeries
lho = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
llo = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)

# We can then call the :meth:`~gwpy.timeseries.TimeSeries.asd` method to
# calculated the amplitude spectral density for each
# `~gwpy.timeseries.TimeSeries`:
lhoasd = lho.asd(4, 2)
lloasd = llo.asd(4, 2)

# We can then :meth:`~FrequencySeries.plot` the spectra using the 'standard'
# colour scheme:

plot = lhoasd.plot(label='LIGO-Hanford', color='gwpy:ligo-hanford')
ax = plot.gca()
ax.plot(lloasd, label='LIGO-Livingston', color='gwpy:ligo-livingston')
ax.set_xlim(10, 2000)
ax.set_ylim(5e-24, 1e-21)
ax.legend(frameon=False, bbox_to_anchor=(1., 1.), loc='lower right', ncol=2)
plot.show()
