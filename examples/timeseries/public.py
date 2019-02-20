#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

I would like to study the gravitational wave strain time-series around the
time of an interesting simulated signal during the last science run (S6).

These data are public, so we can load them directly from the web.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# The `TimeSeries` object has a `classmethod` dedicated to fetching open-access
# data hosted by the LIGO Open Science Center, so we can just import that
# object
from gwpy.timeseries import TimeSeries

# then call the `~TimeSeries.fetch_open_data` method, passing it the prefix
# for the interferometer we want ('L1'), and the GPS start and stop times of
# our query:
data = TimeSeries.fetch_open_data('L1', 968654552, 968654562)

# and then we can make a plot:
plot = data.plot(
    title='LIGO Livingston Observatory data for HW100916',
    ylabel='Strain amplitude',
)
plot.show()
