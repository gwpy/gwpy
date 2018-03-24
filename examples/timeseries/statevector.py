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

"""Plotting segments for a `StateVector`

I would like to examine the state of the internal seismic isolation system supporting the Fabry-Perot mirror at the end of the Y-arm at LHO, in order to investigate a noise source.

These data are private to the LIGO Scientific Collaboration and the Virgo Collaboration, but collaboration members can use the NDS2 service to download data.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we can import the `StateVector` class:
from gwpy.timeseries import StateVector

# Next, we define which bits we want to use, and can then
# :meth:`~StateVector.get` the data:
bits = [
    'Summary state',
    'State 1 damped',
    'Stage 1 isolated',
    'Stage 2 damped',
    'Stage 2 isolated',
    'Master switch ON',
    'Stage 1 WatchDog OK',
    'Stage 2 WatchDog OK',
]

data = StateVector.get('L1:ISI-ETMX_ODC_CHANNEL_OUT_DQ', 'May 22 2014 14:00', 'May 22 2014 15:00', bits=bits)
data = data.astype('uint32')  # hide

# For this example, we wish to :meth:`~StateVector.resample` the data to a
# much lower rate, to make visualising the state much easier:
data = data.resample(16)

# Finally, we make a :meth:`~StateVector.plot`, passing `insetlabels=True` to
# display the bit names inside the axes:
plot = data.plot(insetlabels=True)
plot.gca().set_title('LLO ETMX internal seismic isolation state')
plot.add_bitmask('0b11101110')
plot.show()
