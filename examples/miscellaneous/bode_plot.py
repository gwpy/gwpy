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
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>"""GWpy Example: plotting a time-series

"""Plotting a filter

I would like to look at the Bode representation of a linear filter.
"""

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__currentmodule__ = 'gwpy.plotter'

# First, we import the objects we need
from math import pi
from scipy import signal
from gwpy.plotter import BodePlot

# Now, we can calculate a fourth-order butterworth filter using the
# :meth:`~scipy.signal.butter` function
highpass = signal.butter(4, 10 * (2. * pi), btype='highpass',
                         analog=True)

# The `BodePlot` knows how to plot filters:
plot = BodePlot(highpass)
plot.maxes.set_title('10\,Hz high-pass filter (4th order Butterworth)')
plot.show()
