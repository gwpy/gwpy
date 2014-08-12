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

"""
GWpy Example: plotting a filter
-------------------------------

I would like to look at the Bode representation of a linear filter.
"""

from scipy import signal
from gwpy.plotter import BodePlot

highpass = signal.butter(4, 10 * (2. * signal.np.pi), btype='highpass',
                         analog=True)
plot = BodePlot(highpass)
plot.maxes.set_title('10\,Hz high-pass filter')

if __name__ == '__main__':
    try:
        outfile = __file__.replace('.py', '.png')
    except NameError:
        pass
    else:
        plot.save(outfile)
        print("Example output saved as\n%s" % outfile)
