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

"""Generate the Q-transform of a `TimeSeries`

"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we identify the GPS time of interest:
gps = 968654558

# and use that to define the start and end times of our required data
duration = 32
start = int(round(gps - duration/2.))
end = start + duration

# next, we import the `TimeSeries` and fetch some open data from
# `LOSC <//losc.ligo.org>`_:
from gwpy.timeseries import TimeSeries
data = TimeSeries.fetch_open_data('H1', start, end)

# and next we generate the `~TimeSeries.q_transform` of these data:
qspecgram = data.q_transform()

# Now, we can plot the resulting `~gwpy.spectrogram.Spectrogram`, focusing on a
# specific window around the interesting time
#
# .. note::
#
#    Using `~gwpy.spectrogram.Spectrogram.crop` is highly recommended at
#    this stage because rendering the high-resolution spectrogram as it is
#    done here is very slow (for experts this is because we're using
#    `~matplotlib.axes.Axes.pcolormesh` and not any sort of image
#    interpolation, mainly to support both linear and log scaling nicely)

plot = qspecgram.crop(gps-.3, gps+.1).plot(figsize=[8, 6])
ax = plot.gca()
ax.set_epoch(gps)
ax.set_yscale('log')
ax.set_xlabel('Time [milliseconds]')
ax.set_ylim(50, 1000)
ax.grid(True, axis='y', which='both')
plot.add_colorbar(cmap='viridis', label='Normalized energy')
plot.show()

# I think we just detected a gravitational wave signal! But, before you get too exited, this is an example of a 'blind
# injection', a simulated signal introduced into the interferometer(s) in order to test the detection process end-to-end.
# For more details, see `here <http://www.ligo.org/scientists/GW100916/>`_.
